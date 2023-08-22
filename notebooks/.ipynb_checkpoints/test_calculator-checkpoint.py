# Test cases for Calculator

import calculator

def test_addition():
    assert calculator.addition(2, 3) == 5
    assert calculator.addition(-1, 1) == 0
    assert calculator.addition(0, 0) == 0


def test_subtraction():
    assert calculator.subtraction(5, 3) == 2
    assert calculator.subtraction(10, 5) == 5
    assert calculator.subtraction(0, 0) == 0


if __name__ == '__main__':
    test_addition()
    test_subtraction()
