use std::{marker::PhantomData, sync::Arc};

use crate::{Engine, Error, model::Model};

pub struct TractEngine<In, M: Model<In>> {
    model: Arc<M>,
    _m: PhantomData<In>,
}

impl<In, M: Model<In>> Engine<In> for TractEngine<In, M>
where
    In: Send + 'static,
    M::Out: Send + 'static,
{
    type Out = M::Out;

    async fn infer(
        &self,
        input: impl futures::Stream<Item = Result<In, Error>> + Send + 'static,
    ) -> Result<impl futures::Stream<Item = Result<Self::Out, Error>> + Send + 'static, Error> {
        let _ = input;

        Ok(futures::stream::empty())
    }
}
