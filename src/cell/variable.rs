use crate::cell::function::Function;

pub struct Variable<T, F>
where
    F: Function<T, F>,
{
    pub data: T,
    pub grad: Option<T>,
    pub creater: Option<F>,
}

impl<T, F: Function<T, F>> Variable<T, F> {
    pub fn new(data: T) -> Self {
        Self {
            data: data,
            grad: None,
            creater: None,
        }
    }

    pub fn set_creater(&mut self, func: F) {
        self.creater = Some(func);
    }
}
