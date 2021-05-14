//#ifndef VISION_TENSOR_H
//#define VISION_TENSOR_H
//
//#include <cstring>
//#include <memory>
//#include <string>
//#include <vector>
//
//namespace vision {
//
///// Data type of inference
//enum DType {
//    FP32 = 0,
//    FP16 = 1,
//    INT8 = 2,
//    FP64 = 3,
//    DTYPE_UNKNOWN
//};
//
///// Data layout
//enum DLayout {
//    NCHW = 0,
//    NHWC = 1
//};
//
///// cuda copy kind
//enum CudaMemcpyKind {
//    CUDA_MEMCPY_HOST_TO_HOST          =   0,      /**< Host   -> Host */
//    CUDA_MEMCPY_HOST_TO_DEVICE       =   1,      /**< Host   -> Device */
//    CUDA_MEMCPY_DEVICE_TO_HOST       =   2,      /**< Device -> Host */
//    CUDA_MEMCPY_DEVICE_TO_DEVICE     =   3,      /**< Device -> Device */
//    CUDA_MEMCPY_DEFAULT              =   4
//};
//
///// Basic data structure
//class Tensor {
//public:
//    Tensor();
//    explicit Tensor(int w, DLayout layout = NCHW, DType dtype = FP32);
//    Tensor(int w, int h, DLayout layout = NCHW, DType dtype = FP32);
//    Tensor(int w, int h, int c, DLayout layout = NCHW, DType type = FP32);
//
//    explicit Tensor(int w, DType dtype = FP32, DLayout layout = NCHW);
//    Tensor(int w, int h, DType dtype = FP32, DLayout layout = NCHW);
//    Tensor(int w, int h, int c, DType type = FP32, DLayout layout = NCHW);
//
//    Tensor(int w, void* data, DType dtype = FP32, DLayout layout = NCHW);
//    Tensor(int w, int h, void* data, DType dtype = FP32, DLayout layout = NCHW);
//    Tensor(int w, int h, int c, void* data, DType type = FP32, DLayout layout = NCHW);
//
//    Tensor(int w, void* data, DLayout layout = NCHW, DType dtype = FP32);
//    Tensor(int w, int h, void* data, DLayout layout = NCHW, DType dtype = FP32);
//    Tensor(int w, int h, int c, void* data, DLayout layout = NCHW, DType type = FP32);
//
//    Tensor(const Tensor& t);
//
//    ~Tensor();
//
//    Tensor& operator=(const Tensor& t);
//    Tensor clone() const;
//
//    Tensor change_layout(DLayout layout);
//    Tensor change_dtype(DType dtype);
//
//    void create(int w, DType dtype = FP32, DLayout layout = NCHW);
//    void create(int w, int h, DType dtype = FP32, DLayout layout = NCHW);
//    void create(int w, int h, int c, DType dtype = FP32, DLayout layout = NCHW);
//    void create(int w, DLayout layout = NCHW, DType dtype = FP32);
//    void create(int w, int h, DLayout layout = NCHW, DType dtype = FP32);
//    void create(int w, int h, int c, DLayout layout = NCHW, DType dtype = FP32);
//    void release();
//
//    bool empty() const;
//    size_t size() const;
//    size_t len() const;
//    void set_name(const std::string& name);
//    std::string get_name() const;
//    int get_ref_count() const;
//
//    int w;
//    int h;
//    int c;
//    int stride;
//    int dims;
//    void* data;
//    DType dtype;
//    DLayout layout;
//
//private:
//    void add_ref() const;
//    std::string _name;
//    int* _ref_count;
//};
//
//using TensorArray = std::vector<Tensor>;
//using TensorPtr = std::shared_ptr<Tensor>;
//
//} // namespace vision
//
//#endif //VISION_TENSOR_H
