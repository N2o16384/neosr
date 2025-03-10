import warnings
from copy import deepcopy
from os import path as osp, remove
from io import BytesIO
import tempfile
from shutil import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnxconverter_common.float16 import convert_float_to_float16
from onnxconverter_common import auto_mixed_precision

# from onnxsim import simplify
from neosr.archs import build_network
from neosr.utils.options import parse_options


def load_net():
    # build_network
    print(f"\n-------- Attempting to build network [{args.network}].")
    if args.network is None:
        msg = "Please select a network using the -net option"
        raise ValueError(msg)
    net_opt = {"type": args.network}

    if args.network == "omnisr":
        net_opt["upsampling"] = args.scale
        net_opt["window_size"] = args.window
    
    to_dict = lambda s: {
        k: (
            int(v) if v.isdigit()
            else float(v) if v.replace('.', '').isdigit() and '.' in v
            else True if v.lower() == 'true'
            else False if v.lower() == 'false'
            else v
        )
        for k, v in (
            pair.split('=', 1)
            for pair in s
            if '=' in pair
        )
    }
    if args.netconf:
        net_config=to_dict(args.netconf)
        for k, v in net_config.items():
            net_opt[k] = v

    if args.window:
        net_opt["window_size"] = args.window

    net = build_network(net_opt)
    load_net = torch.load(
        args.input, map_location=torch.device("cuda"), weights_only=True
    )
    # find parameter key
    print("-------- Finding parameter key...")
    param_key: str | None = None
    try:
        if "params-ema" in load_net:
            param_key = "params-ema"
        elif "params" in load_net:
            param_key = "params"
        elif "params_ema" in load_net:
            param_key = "params_ema"
        load_net = load_net[param_key]
    except:  # noqa: S110
        pass

    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith("module."):
            load_net[k[7:]] = v
            load_net.pop(k)

    # load_network and send to device and set to eval mode
    net.load_state_dict(load_net, strict=True)  # type: ignore[reportAttributeAccessIssue,attr-defined]
    net.eval()

    # plainusr
    try:
        for module in net.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy(args.prune)
        print("-------- Reparametrization completed successfully.")
    except:  # noqa: S110
        pass

    net = net.to(device="cuda", non_blocking=True)  # type: ignore[reportAttributeAccessIssue,attr-defined]
    print(f"-------- Successfully loaded network [{args.network}].")
    torch.cuda.empty_cache()

    return net


def assert_verify(onnx_model, torch_model, isMixed = False) -> None:
    dyinput = 0 if isMixed == True else 1
    if args.static is not None:
        dummy_input = torch.randn(dyinput, *args.static, requires_grad=True)
    else:
        dummy_input = torch.randn(dyinput, 3, 20, 20, requires_grad=True)
    # onnxruntime output prediction
    # NOTE: "CUDAExecutionProvider" errors if some nvidia libs
    # are not found, defaulting to cpu
    ort_session = onnxruntime.InferenceSession(
        onnx_model, providers=["CPUExecutionProvider"]
    )
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # torch outputs
    with torch.inference_mode():
        torch_outputs = torch_model(dummy_input)

    # final assert - default tolerance values - rtol=1e-03, atol=1e-05
    np.testing.assert_allclose(
        torch_outputs.detach().cpu().numpy(), ort_outs[0], rtol=0.01, atol=0.001
    )
    print("-------- Model successfully verified.")


def to_onnx() -> None:
    # error if network can't be converted
    net_error = ["craft", "ditn"]
    if args.network in net_error:
        msg = f"Network [{args.network}] cannot be converted to ONNX."
        raise RuntimeError(msg)

    # load network and send to device
    model = load_net()
    # set model to eval mode
    model.eval()

    # set static or dynamic
    if args.static is not None:
        dummy_input = torch.randn(1, *args.static, requires_grad=True)
    else:
        dummy_input = torch.randn(1, 3, 20, 20, requires_grad=True)

    # dict for dynamic axes
    if args.static is None:
        dyn_axes = {
            "dynamic_axes": {
                "input": {0: "batch_size", 2: "width", 3: "height"},
                "output": {0: "batch_size", 2: "width", 3: "height"},
            },
            "input_names": ["input"],
            "output_names": ["output"],
        }
    else:
        dyn_axes = {"input_names": ["input"], "output_names": ["output"]}

    # add _fp32 suffix to output str
    if args.output:
        output_prefix = osp.splitext(args.output)[0]
    else:
        output_prefix = osp.splitext(args.input)[0]
    # begin conversion
    print("-------- Starting ONNX conversion (this can take a while)...")

    output_onnx = output_prefix + '.onnx'
    model_buffer = BytesIO()
    with torch.inference_mode(), torch.device("cpu"):
        # TODO: add param dynamo=True as a switch
        # py2.5 supports the verify=True flag now as well
        torch.onnx.export(
            model,
            dummy_input,
            model_buffer,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=False,
            **(dyn_axes or {}),  # type: ignore
        )
    print("-------- ONNX model was successful to IOBytes. Validating...")
    model_buffer.seek(0)
    onnx_model = onnx.load(model_buffer)
    try:
        onnx.checker.check_model(onnx_model)
        print("-------- Validation is successful...")
    except Exception as e:
        print("-------- Validation is failed...")
        return

    if args.optimize:
        model_buffer_optimized = BytesIO()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            model_temp_path = temp_file.name
        session_opt = onnxruntime.SessionOptions()
        session_opt.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        session_opt.optimized_model_filepath = model_temp_path
        onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), session_opt
        )
        with open(model_temp_path, 'rb') as f:
            optimized_model_bytes = f.read()
        remove(model_temp_path)
        model_buffer_optimized.seek(0)
        model_buffer_optimized.write(optimized_model_bytes)
        print("-------- Optimized model was successful to IOBytes. Validating...")
        model_buffer_optimized.seek(0)
        onnx_model_optimized = onnx.load(model_buffer_optimized)
        try:
            onnx.checker.check_model(onnx_model_optimized)
            print("-------- Validation is successful...")
        except Exception as e:
            print("-------- Validation is failed...")

    if args.fp16:
        print("-------- Converting to Fp16...")
        if args.optimize:
            to_fp16 = convert_float_to_float16(onnx_model_optimized)  # type: ignore[reportPossiblyUnboundVariable]
        else:
            to_fp16 = convert_float_to_float16(onnx_model)
        # save
        onnx.save(to_fp16, output_onnx)
        print(
            f"-------- Model successfully converted to Fp16(half)-precision. Saved at: {output_onnx}."
        )
    elif args.fpmix:
        print("-------- Converting to Fp32/16 mixed...")
        if args.static is not None:
            dummy_dyinput = torch.randn(0, *args.static, requires_grad=True)
        else:
            dummy_dyinput = torch.randn(0, 3, 20, 20, requires_grad=True)
        onnx_model_final = onnx_model_optimized if args.optimize else onnx_model
        ort_session = onnxruntime.InferenceSession(
            onnx_model_final.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_dyinput.detach().cpu().numpy()}
        to_fpmix = auto_mixed_precision.auto_convert_mixed_precision(onnx_model_final, ort_inputs, rtol=0.99, atol=0.999, keep_io_types=True)
        onnx.save(to_fpmix, output_onnx)
        print(
            f"-------- Model successfully converted to Mixed-precision. Saved at: {output_onnx}."
        )
    else:
        print("-------- Converting to Fp32...")
        onnx_model_final = onnx_model_optimized if args.optimize else onnx_model
        onnx.save(onnx_model_final, output_onnx)
        print(
            f"-------- Model successfully converted to Fp32-precision. Saved at: {output_onnx}."
        )

    if args.nocheck is False:
        print("-------- Verify model precision...")
        if args.fpmix:
            assert_verify(output_onnx, model, isMixed = True)
        else:
            assert_verify(output_onnx, model)

    return
        
    if args.fulloptimization:
        msg = "ONNXSimplify has been temporarily disabled."
        raise ValueError(msg)
        """
        # error if network can't run through onnxsim
        opt_error = ["omnisr"]
        if args.network in opt_error:
            msg = f"Network [{args.network}] doesnt support full optimization."
            raise RuntimeError(msg)

        print("-------- Running full optimization (this can take a while)...")

        # run onnxsim
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            model_temp_path = temp_file.name
        copy(output_onnx, model_temp_path)
        simplified, check = simplify(onnx.load(model_temp_path))
        assert check, "Couldn't validate ONNX model."

        # save and verify
        onnx.save(simplified, output_onnx)
        remove(model_temp_path)
        print(
            f"-------- Model successfully optimized. Saved at: {output_fulloptimized}\n"
        )
        """


if __name__ == "__main__":
    torch.set_default_device("cuda")
    warnings.filterwarnings("ignore", category=UserWarning)
    root_path = Path(Path(__file__) / osp.pardir).resolve()
    __, args = parse_options(str(root_path))
    to_onnx()
