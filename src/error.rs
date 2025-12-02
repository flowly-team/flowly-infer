#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("async cuda error: {0}")]
    CudaError(#[from] async_cuda::Error),

    #[error("async tensorrt error: {0}")]
    TensorrtError(#[from] async_tensorrt::Error),

    #[error("image error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("internal error")]
    InternalError,

    #[error("unknown quantization mode: {0}")]
    UnknownQuantizationMode(String),

    #[error("no predefined execution contexts")]
    NoExecutionContext,
}
