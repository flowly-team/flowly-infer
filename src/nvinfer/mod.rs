pub mod engine;
pub mod stream;

pub use engine::{Config, Session};

use crate::{Error, model::Model};

pub async fn nvinfer<I, M>(model: M, config: Config) -> Result<Session<I, M>, Error>
where
    I: Send + 'static,
    M: Model<I> + 'static,
    M::Out: 'static,
{
    Session::new(model, config).await
}
