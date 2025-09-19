
from base.memory.scoring import assess_importance

def test_importance_basic():
    assert assess_importance("remember this: my sister is Alice") >= 50
    assert assess_importance("hello") < 25