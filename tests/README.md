# Qibolab Tests

Here are some instructions on how to execute tests
with or without QPU access.

## Tests without QPU

Tests can be executed on any machine by executing
```sh
pytest
```

This will execute all tests that are **not** marked
with the `qpu` marker, using the platforms in the
`tests/dummy_qrc` folder.

## Tests on QPU

Tests marked using `@pytest.mark.qpu` require access
to control electronics. These can be executed using
```sh
pytest -m qpu --platform {my_platform_name}
```

Where `{my_platform_name}` is the name of the platform
to execute tests on. This platform should be defined
in a ``create()`` method that is saved in a separate
Python module, as explained in the qibolab documentation.
Qibolab should be pointed to the directory that contains
the platform creation modules via the `QIBOLAB_PLATFORMS`
environment variable.

When running tests on QPU, the instruments that are available
in the given platform are known and the related tests are
executed or skipped automatically.

[qibolab_platforms_qrc](https://github.com/qiboteam/qibolab_platforms_qrc) provides some examples on how to
define different platforms.

Note that the `--platform` option used in the `pytest`
command above specifies the platforms used by `qpu` tests.
Non-qpu tests will still use the platforms under `tests/dummy_qrc` regardless of the `--platform` option and the
value of `QIBOLAB_PLATFORMS` environment variable.
