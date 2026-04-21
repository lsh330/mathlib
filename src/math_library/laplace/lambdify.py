# lambdify.py — AST to_string() 기반 Python callable 변환 (Phase C)
# to_string() 출력을 Python 수식으로 변환 후 exec으로 컴파일

from __future__ import annotations
from typing import List, Callable


# ================================================================== 수식 파서
# to_string() 형식:
#   숫자: 정수 또는 부동소수 (예: 3, -2.0, 1/3 아님 → 소수)
#   변수: 이름 문자열
#   Sum:  (a + b + c)  또는  (a + b)
#   Mul:  (a*b*c)      또는  (a*b)
#   Pow:  (base**exp)
#   Neg:  (-x)
#   Func: sin(arg), cos(arg), exp(arg), ...
#
# 정규화 규칙 (pool 출력에서 관찰):
#   - 외부 괄호: Sum/Mul/Pow는 항상 (...)로 감쌈
#   - 분수: Mul([num, Pow(den, -1)]) → (num * den**-1)  또는  num*den**-1
#   - 곱셈: (a**-1)*2 → (s + 1)**-1*2 형태 (Mul 순서 가변)


def _find_matching_paren(s: str, start: int) -> int:
    """s[start] = '(' 기준으로 매칭되는 ')' 위치 반환."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_top_level(s: str, sep: str) -> list:
    """괄호 외부 레벨에서 sep 문자로 분리."""
    parts = []
    depth = 0
    current = []
    i = 0
    n = len(sep)
    while i < len(s):
        if s[i] == '(':
            depth += 1
            current.append(s[i])
            i += 1
        elif s[i] == ')':
            depth -= 1
            current.append(s[i])
            i += 1
        elif depth == 0 and s[i:i+n] == sep:
            parts.append(''.join(current))
            current = []
            i += n
        else:
            current.append(s[i])
            i += 1
    parts.append(''.join(current))
    return parts


def _convert(s_repr: str, var_names: List[str], backend: str) -> str:
    """
    to_string() 형식 문자열을 Python 수식으로 변환.
    재귀적으로 처리.
    """
    s = s_repr.strip()
    if not s:
        return '0'

    # 백엔드 접두어
    _m = {'numpy': 'np', 'cmath': 'cmath', 'math': 'math'}.get(backend, 'math')

    # ── 수치 리터럴 (정수/부동소수)
    # 음수 포함: -3.5
    try:
        v = float(s)
        # 정수화 가능한 경우
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v)
    except ValueError:
        pass

    # ── 변수
    if s in var_names:
        return s

    # ── 함수 호출: name(args) — Func 노드
    func_map = {
        'sin':   f'{_m}.sin',
        'cos':   f'{_m}.cos',
        'tan':   f'{_m}.tan',
        'arcsin': f'{_m}.asin',
        'arccos': f'{_m}.acos',
        'arctan': f'{_m}.atan',
        'sinh':  f'{_m}.sinh',
        'cosh':  f'{_m}.cosh',
        'tanh':  f'{_m}.tanh',
        'exp':   f'{_m}.exp',
        'ln':    f'{_m}.log',
        'log':   f'{_m}.log10' if backend != 'cmath' else f'{_m}.log',
        'sqrt':  f'{_m}.sqrt',
    }
    for fname, pyname in func_map.items():
        if s.startswith(fname + '(') and s.endswith(')'):
            inner = s[len(fname)+1:-1]
            return f"{pyname}({_convert(inner, var_names, backend)})"

    # Heaviside: heaviside(arg) — Python inline lambda
    if s.startswith('heaviside(') and s.endswith(')'):
        inner = s[len('heaviside('):-1]
        inner_py = _convert(inner, var_names, backend)
        # u(x): x<0→0, x=0→0.5, x>0→1
        return f'(0.0 if ({inner_py}) < 0 else (0.5 if ({inner_py}) == 0 else 1.0))'

    # Dirac: dirac(arg) — 수치 평가 불가, 0 반환
    if s.startswith('dirac(') and s.endswith(')'):
        return '0.0'

    # ── 괄호로 감싸인 표현식
    if s.startswith('(') and s.endswith(')'):
        # 매칭 괄호 확인 (전체가 하나의 괄호 쌍인지)
        end = _find_matching_paren(s, 0)
        if end == len(s) - 1:
            inner = s[1:-1]
            return _convert_inner(inner, var_names, backend, _m)

    # ── 괄호 없는 복합 수식 처리
    return _convert_inner(s, var_names, backend, _m)


def _convert_inner(s: str, var_names: List[str], backend: str, _m: str) -> str:
    """괄호를 제거한 내부 수식 변환."""

    # ── ** (거듭제곱) 분리: 최외부 레벨에서 ** 위치 탐색
    # Pow 우선순위가 높으므로 ** 를 먼저 처리하지 않고
    # 덧셈 → 곱셈 → 거듭제곱 순으로 처리 (우선순위 낮은 것부터)

    # ── + 분리 (Sum)
    parts = _split_top_level(s, ' + ')
    if len(parts) > 1:
        py_parts = [_convert(p.strip(), var_names, backend) for p in parts]
        return '(' + ' + '.join(py_parts) + ')'

    # ── * 분리 (Mul)
    # 단, ** 는 * 로 분리되지 않도록 주의
    parts = _split_mul(s)
    if len(parts) > 1:
        py_parts = []
        for p in parts:
            p = p.strip()
            py_parts.append(_convert(p, var_names, backend))
        # 분모 인수 (음의 지수) 분리: a * b**-n → a / b**n
        return _combine_mul_parts(py_parts, parts, var_names, backend)

    # ── ** (Pow) 분리
    pow_pos = _find_last_pow(s)
    if pow_pos >= 0:
        base_s = s[:pow_pos].strip()
        exp_s  = s[pow_pos+2:].strip()
        base_py = _convert(base_s, var_names, backend)
        exp_py  = _convert(exp_s, var_names, backend)
        # 음의 지수 → 분수 변환 (0을 base로 쓰지 않도록)
        try:
            exp_v = float(exp_s)
            if exp_v < 0:
                # 1 / (base ** abs(exp))
                abs_exp = -exp_v
                if abs_exp == 1.0:
                    return f'(1.0 / {base_py})'
                else:
                    abs_exp_py = str(int(abs_exp)) if abs_exp == int(abs_exp) else repr(abs_exp)
                    return f'(1.0 / ({base_py} ** {abs_exp_py}))'
        except (ValueError, TypeError):
            pass
        return f'({base_py} ** {exp_py})'

    # ── 음수 부정: 선행 -
    if s.startswith('-'):
        inner = s[1:].strip()
        return f'(-{_convert(inner, var_names, backend)})'

    # ── 변수 (함수 없는 이름)
    if s.isidentifier() or all(c.isalnum() or c in ('_',) for c in s):
        if s in var_names:
            return s
        # 숫자가 아닌 이름 → 그대로 (런타임 변수)
        return s

    # ── 폴백
    return s


def _split_mul(s: str) -> list:
    """최외부 레벨에서 '*' 분리 (** 는 분리하지 않음)."""
    parts = []
    depth = 0
    current = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '(':
            depth += 1
            current.append(ch)
            i += 1
        elif ch == ')':
            depth -= 1
            current.append(ch)
            i += 1
        elif ch == '*' and depth == 0:
            # ** 는 건너뜀
            if i + 1 < len(s) and s[i+1] == '*':
                current.append('*')
                current.append('*')
                i += 2
            else:
                parts.append(''.join(current))
                current = []
                i += 1
        else:
            current.append(ch)
            i += 1
    parts.append(''.join(current))
    return parts


def _find_last_pow(s: str) -> int:
    """최외부 레벨에서 마지막 ** 위치 탐색."""
    depth = 0
    pos = -1
    i = 0
    while i < len(s):
        if s[i] == '(':
            depth += 1
            i += 1
        elif s[i] == ')':
            depth -= 1
            i += 1
        elif s[i] == '*' and i + 1 < len(s) and s[i+1] == '*' and depth == 0:
            pos = i
            i += 2
        else:
            i += 1
    return pos


def _combine_mul_parts(py_parts: list, raw_parts: list, var_names: List[str], backend: str) -> str:
    """
    Mul 인수 목록 → Python 수식.
    음의 지수 인수를 분모로 이동.
    예: [(s+2)**-1, 2] → 2 / (s+2)
    """
    # 단순 곱셈 (음수 지수 감지 없이)
    # 각 부분을 이미 _convert로 변환한 py_parts 사용
    return '(' + ' * '.join(py_parts) + ')'


# ================================================================== lambdify 메인 API

def lambdify(expr, var_names: List[str], backend: str = 'cmath') -> Callable:
    """
    PyExpr AST를 Python callable로 변환.

    Parameters
    ----------
    expr      : PyExpr
    var_names : 변수 이름 리스트 (예: ['s'], ['t'])
    backend   : 'math' | 'cmath' | 'numpy'

    Returns
    -------
    callable
    """
    # AST → Python source
    expr_str = str(expr)
    py_source = _convert(expr_str, list(var_names), backend)

    # 임포트 헤더
    import_map = {
        'math':   'import math',
        'cmath':  'import cmath',
        'numpy':  'import numpy as np',
    }
    import_line = import_map.get(backend, 'import math')

    # 함수 정의
    args = ', '.join(var_names)
    code = f"""{import_line}
def _lambdified_func({args}):
    return {py_source}
"""

    ns: dict = {}
    try:
        exec(compile(code, '<lambdify>', 'exec'), ns)
    except SyntaxError as e:
        raise ValueError(
            f"lambdify: generated invalid Python code:\n{code}\nError: {e}"
        )

    return ns['_lambdified_func']
