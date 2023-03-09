import pytest

import toml
from pathlib import Path

# Agregamos algunas opciones para casos específicos
def pytest_addoption(parser):
    # Algunos test basados en propiedades estadísticas pueden fallar
    # al menos hasta que logre tunearlos. Con este flag se pueden activar, pero
    # por defecto no se corren.
    parser.addoption("--stat_properties", action="store_true",
                     help="run the tests only in case of that command line (marked with marker @stat_properties)")

    # Esta opción está por si en algún momento tenemos más de un archivo de
    # tests para comparar con hector, que puede ser que sea uno más exhaustivo
    # que el otro por ejemplo.
    # De momento, dejando el default está bien.
    parser.addoption(
        "--hector_test_cases",
        action="store",
        default="test_cases.toml",
        help="explicitly provide hector test cases file"
    )

# Este fixture es bobo, solamente para marcar que una función va a correr sobre
# los test cases leidos desde el toml.
@pytest.fixture()
def hector_test_cases():
    pass

def pytest_generate_tests(metafunc):
    # En caso de que un test pida los test cases
    if "hector_test_cases" in metafunc.fixturenames:
        # Pedimos el path al archivo de casos
        filename = metafunc.config.getoption("hector_test_cases")

        if not Path(filename).is_file():
            pytest.skip(f"El archivo de casos {filename} no existe (se crea con create_test_cases.py)")

        # Cargamos los casos
        with open(filename,'r') as f:
            casos = toml.load(f)

        casos_list = []

        for modelo,sub_casos in casos.items():
            for etiqueta,parametros in sub_casos.items():
                casos_list.append( (modelo,etiqueta,parametros) )

        # Se parametriza la función con estos tres parámetros.
        # (el tercero es el diccionario de opciones).
        # Implicitamente, todo test que tenca hector_test_cases tiene que pedir
        # también modelo, etiqueta, parámetros.
        metafunc.parametrize("modelo, etiqueta, parametros", casos_list)

def pytest_runtest_setup(item):
    # Saltear por defauld los tests de stat_properties
    if 'stat_properties' in item.keywords and not item.config.getoption("--stat_properties"):
        pytest.skip("need --stat_properties option to run this test")
