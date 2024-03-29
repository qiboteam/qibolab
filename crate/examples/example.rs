use anyhow::Result;
use qibolab::execute_qasm;

const CODE: &str = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg a[2];
cz q[0],q[2];
gpi2(0.3) q[1];
cz q[1],q[2];
measure q[0] -> a[0];
measure q[2] -> a[1];
"#;

fn main() -> Result<()> {
    let circuit = CODE.to_owned();
    let platform = "dummy".to_owned();
    let nshots = 10;

    println!(
        "---\nExecuting:\n'''{}'''\n\non: {}\nwith: {} shots\n---\n",
        circuit, platform, nshots
    );

    let res = execute_qasm(circuit, platform, nshots)?;
    println!("\n{:?}", res);

    Ok(())
}
