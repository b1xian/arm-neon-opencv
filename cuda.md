### cuda 内存管理

#### 1 device内存管理
**cudaMalloc cudaMemcpy**
申请设备内存和向设备内存拷贝数据。  
cudaMalloc和cudaMemcpy的速度(ms)  
malloc cost: 0.007200  
copy cost: 2.406400  
cuda malloc cost: 0.092288  
cuda copy cost: 0.003072  

#### 2 页锁定的host内存 
**cudaHostAlloc( void\*\*, size, cudaHostAllocDefault )**  
对于cudaMemcpy调用的宿主内存，可以使用cudaHostAlloc来分配页锁定物理内存。  
但事实上，cudaHostAlloc的速度大幅度慢于malloc，只是在cudaMemcpy时，
其拷贝速度略高于malloc的宿主内存。  
因此并不推荐使用cudaHostAlloc来做宿主机的内存管理。

#### 3 零拷贝宿主机内存 
**cudaHostAlloc( void\*\*, size, cudaHostAllocWriteCombined | cudaHostAllocMapped )**  
**cudaHostGetDevicePointer(void\*\* device_ptr, void \*host_ptr, int flags)**
cudaHostGetDevicePointer可以直接获取cudaHostAlloc申请的内存在GPU上的指针，
从而避免GPU的内存申请和拷贝操作。  
但由于cudaHostAlloc申请的是宿主机内存，核函数是对CPU内存的运算，效率会因此大幅度降低。

