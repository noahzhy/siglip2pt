import onnxruntime as ort
import numpy as np
import time
import argparse
import concurrent.futures
import threading
import os


def get_args():
    parser = argparse.ArgumentParser(description="ONNX Runtime Speed Test")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    # Default threads to CPU count, but at least 1
    default_threads = max(os.cpu_count() or 4, 1)
    parser.add_argument("--threads", type=int, default=default_threads, help="Number of concurrent threads for requests")
    parser.add_argument("--intra_op_threads", type=int, default=8, help="Intra-op threads (default 1 for high throughput)")
    parser.add_argument("--inter_op_threads", type=int, default=8, help="Inter-op threads (default 1)")
    parser.add_argument("--provider", type=str, default="cuda", choices=["cpu", "cuda", "tensorrt"], help="Execution provider")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size override for dynamic axes")
    parser.add_argument("--use_io_binding", action="store_true", help="Use IOBinding to pre-copy inputs to GPU (measures pure inference speed)")
    parser.add_argument("--enable_cuda_graph", action="store_true", help="Enable CUDA Graph (reduces CPU launch overhead)")
    return parser.parse_args()


def create_session(args):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.intra_op_threads
    sess_options.inter_op_num_threads = args.inter_op_threads
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Enable memory pattern optimization
    sess_options.enable_mem_pattern = True
    
    # Enable cpu memory arena
    sess_options.enable_cpu_mem_arena = True
    
    # Set execution mode
    if args.threads > 1 and args.intra_op_threads == 1:
         sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
         sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = []
    if args.provider == "cuda":
        cuda_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024, # 8GB limit example
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
            'cudnn_conv_use_max_workspace': '1',
        }
        if args.enable_cuda_graph:
            cuda_options['enable_cuda_graph'] = '1'
            
        providers = [
            ('CUDAExecutionProvider', cuda_options),
            'CPUExecutionProvider',
        ]
    elif args.provider == "tensorrt":
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': True,
            }),
            ('CUDAExecutionProvider', {}),
            'CPUExecutionProvider',
        ]
    else:
        providers = ['CPUExecutionProvider']

    print(f"Creating session with providers: {providers}")
    session = ort.InferenceSession(args.model, sess_options, providers=providers)
    
    active_providers = session.get_providers()
    print(f"Active providers: {active_providers}")
    if args.provider == 'cuda' and 'CUDAExecutionProvider' not in active_providers:
        print("\033[93mWARNING: CUDAExecutionProvider requested but not active! Running on CPU?\033[0m")
        
    return session


def generate_inputs(session, batch_size):
    inputs = {}
    for node in session.get_inputs():
        name = node.name
        shape = node.shape
        dtype = node.type
        
        new_shape = []
        for i, dim in enumerate(shape):
            if isinstance(dim, str) or dim is None:
                # Assume the first dynamic dimension is batch size
                if i == 0:
                    new_shape.append(batch_size)
                else:
                    # For other dynamic dimensions, use a default value (e.g., 224 for images, 128 for seq)
                    # This is a heuristic and might need adjustment for specific models
                    new_shape.append(224) 
            else:
                new_shape.append(dim)
        
        print(f"Input: {name}, Original Shape: {shape}, Test Shape: {new_shape}, Type: {dtype}")

        if 'float' in dtype:
            data = np.random.rand(*new_shape).astype(np.float32)
        elif 'int64' in dtype:
            data = np.random.randint(0, 100, size=new_shape).astype(np.int64)
        elif 'int32' in dtype:
            data = np.random.randint(0, 100, size=new_shape).astype(np.int32)
        elif 'bool' in dtype:
            data = np.random.choice([True, False], size=new_shape).astype(np.bool_)
        else:
            print(f"Warning: Unsupported dtype {dtype} for input {name}, using float32")
            data = np.random.rand(*new_shape).astype(np.float32)
            
        inputs[name] = data
    return inputs


def benchmark(args):
    try:
        session = create_session(args)
    except Exception as e:
        print(f"Failed to create session: {e}")
        return

    try:
        inputs = generate_inputs(session, args.batch_size)
    except Exception as e:
        print(f"Failed to generate inputs: {e}")
        return
    
    print("-" * 50)
    print(f"Model: {args.model}")
    print(f"Provider: {args.provider}")
    print(f"Concurrent Threads: {args.threads}")
    print(f"Intra-op Threads: {args.intra_op_threads}")
    print(f"Inter-op Threads: {args.inter_op_threads}")
    print("-" * 50)

    if args.enable_cuda_graph and args.threads > 1:
        print("\033[93mWARNING: CUDA Graph enabled. Forcing threads=1 as CUDA Graph does not support concurrent execution with different buffers.\033[0m")
        args.threads = 1

    # Warmup
    print(f"Warming up for {args.warmup} iterations...")
    if args.use_io_binding:
        # Create a temporary binding for warmup to ensure graph capture happens if needed
        # Note: Graph might be recaptured in worker if buffers change, but this warms up the session.
        warmup_binding = session.io_binding()
        for name, arr in inputs.items():
            device = 'cuda' if args.provider == 'cuda' else 'cpu'
            ort_val = ort.OrtValue.ortvalue_from_numpy(arr, device, 0)
            warmup_binding.bind_ortvalue_input(name, ort_val)
        for out in session.get_outputs():
            device = 'cuda' if args.provider == 'cuda' else 'cpu'
            warmup_binding.bind_output(out.name, device)
            
        for _ in range(args.warmup):
            session.run_with_iobinding(warmup_binding)
            warmup_binding.copy_outputs_to_cpu()
    else:
        for _ in range(args.warmup):
            session.run(None, inputs)
    
    print(f"Starting benchmark for {args.duration} seconds...")
    
    start_time = time.perf_counter()
    end_time = start_time + args.duration
    
    request_count = 0
    latencies = []
    lock = threading.Lock()
    
    def worker():
        nonlocal request_count
        local_count = 0
        local_latencies = []
        
        # Setup IOBinding if requested
        binding = None
        if args.use_io_binding:
            binding = session.io_binding()
            # Bind inputs (assuming inputs are on CPU initially, copy to device)
            for name, arr in inputs.items():
                # Move data to device (e.g. CUDA) if provider is CUDA
                device = 'cuda' if args.provider == 'cuda' else 'cpu'
                ort_val = ort.OrtValue.ortvalue_from_numpy(arr, device, 0)
                binding.bind_ortvalue_input(name, ort_val)
            
            # Bind outputs to device (avoid copy back to host for pure throughput test)
            # Note: This measures kernel execution time mostly.
            device = 'cuda' if args.provider == 'cuda' else 'cpu'
            for out in session.get_outputs():
                binding.bind_output(out.name, device)

        while time.perf_counter() < end_time:
            t0 = time.perf_counter()
            try:
                if binding:
                    session.run_with_iobinding(binding)
                    # Synchronize by copying outputs to CPU. 
                    # Without this, run_with_iobinding is async on GPU and we measure submission time, not execution time.
                    binding.copy_outputs_to_cpu()
                else:
                    session.run(None, inputs)
                t1 = time.perf_counter()
                local_latencies.append((t1 - t0) * 1000) # ms
                local_count += 1
            except Exception as e:
                print(f"Error during inference: {e}")
                break
        
        with lock:
            request_count += local_count
            latencies.extend(local_latencies)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(worker) for _ in range(args.threads)]
        concurrent.futures.wait(futures)
        
    total_time = time.perf_counter() - start_time
    
    if request_count == 0:
        print("No requests completed.")
        return

    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    qps = request_count / total_time
    
    print("\nResults:")
    print(f"Total Requests: {request_count}")
    print(f"Total Time: {total_time:.2f} s")
    print(f"QPS: {qps:.2f}")
    print(f"Avg Throughput: {qps / args.threads:.2f} QPS/thread")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"P50 Latency: {p50:.2f} ms")
    print(f"P90 Latency: {p90:.2f} ms")
    print(f"P99 Latency: {p99:.2f} ms")


if __name__ == "__main__":
    args = get_args()
    benchmark(args)
