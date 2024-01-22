use pyo3::prelude::*;
use pyo3::types::PyDict;

pub fn execute_qasm(circuit: String, platform: String, nshots: u32) -> PyResult<Vec<u32>> {
    // TODO: move to the example, here for debug
    println!(
        "---\nExecuting:\n'''{}'''\n\non: {}\nwith: {} shots\n---\n",
        circuit, platform, nshots
    );

    Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("circuit", circuit)?;
        kwargs.set_item("platform", platform)?;
        kwargs.set_item("nshots", nshots)?;

        let qibolab = PyModule::import(py, "qibolab")?;
        qibolab
            .getattr("execute_qasm")?
            .call((), Some(kwargs))?
            .call_method0("samples")?
            .call_method0("ravel")?
            .extract()
    })
}
