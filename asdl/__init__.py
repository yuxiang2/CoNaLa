import six

if six.PY2:
    from .lang.py.py_transition_system import PythonTransitionSystem
else:
    from .lang.py3.py3_transition_system import Python3TransitionSystem
    # from asdl.lang.sql.sql_transition_system import SqlTransitionSystem

