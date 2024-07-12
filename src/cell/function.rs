pub mod exp;
pub mod power;
pub mod square;

use ndarray::ArrayD;

use crate::cell::variable::Variable;

pub trait Function<T, F> {
    fn call(&mut self, input: &Variable<ArrayD<T>, F>) -> Variable<ArrayD<T>, F> {
        let x = &input.data;
        let y = self.forward(x);
        let output = Variable::new(y);
        self.remember_input(input);
        output
    }
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T>;
    fn backward(&mut self, gy: &ArrayD<T>) -> ArrayD<T>;
    fn remember_input(&mut self, input: &Variable<ArrayD<T>, F>);
    fn remember_output(&mut self, output: &Variable<ArrayD<T>, F>);
}
