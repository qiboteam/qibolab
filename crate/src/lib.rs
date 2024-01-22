use ndarray::Array2;
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub fn execute_qasm(circuit: String, platform: String, nshots: u32) -> PyResult<Array2<i32>> {
    Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("circuit", circuit)?;
        kwargs.set_item("platform", platform)?;
        kwargs.set_item("nshots", nshots)?;

        let qibolab = PyModule::import(py, "qibolab")?;
        let pyarray: &PyArray2<i32> = qibolab
            .getattr("execute_qasm")?
            .call((), Some(kwargs))?
            .call_method0("samples")?
            .extract()?;

        Ok(pyarray.to_owned_array())
    })
}
