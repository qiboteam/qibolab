use pyo3::prelude::*;

pub fn execute_qasm(circuit: String, platform: String, nshots: u32) -> PyResult<Vec<u32>> {
    // TODO: move to the example, here for debug
    println!(
        "---\nExecuting:\n'''{}'''\n\non: {}\nwith: {} shots\n---\n",
        circuit, platform, nshots
    );

    Python::with_gil(|py| {
        let qibolab = PyModule::import(py, "qibolab")?;
        qibolab
            .getattr("execute_qasm")?
            .call1((circuit, platform, nshots))?
            .extract()
    })
}
