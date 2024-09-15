import torch, time

device = "cuda" if torch.cuda.is_available() else "cpu"

tensor = torch.tensor([1, 2, 3]) # By default, it will be created on the CPU
print(tensor.device)

tensor_gpu = tensor.to(device) # Now it will be moved to the GPU if available
print(tensor_gpu.device)

tensor_back_to_cpu = tensor_gpu.to("cpu") # Now it will be moved back to the CPU
print(tensor_back_to_cpu.device)




matrix_A = torch.rand(10000,10000)
matrix_B = torch.rand(10000,10000)

start_t = time.time()
result = torch.matmul(matrix_A, matrix_B)
end_t = time.time()
print("Time taken for matrix multiplication: ", end_t - start_t)


matrix_A = matrix_A.to(device)
matrix_B = matrix_B.to(device)

start_t = time.time()
result = torch.matmul(matrix_A, matrix_B)
end_t = time.time()
print("Time taken for matrix multiplication on GPU: ", end_t - start_t)