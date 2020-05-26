#!/usr/bin/env python3


import functools
import mpmath as mp
import os
import readline
import rply
import sys


def compose(f, g):
    return lambda *x: f(g(*x))


def printIndented(*things):
    for thing in things:
        for line in str(thing).splitlines():
            print('    %s' % line)


def findByPrefix(prefix, strings):
    return [s for s in strings if s.startswith(prefix)]


def degrees(x):
    if isinstance(x, mp.mpc):
        return mp.degrees(mp.re(x)) + mp.im(x) * mp.j
    return mp.degrees(x)


def radians(x):
    if isinstance(x, mp.mpc):
        return mp.radians(mp.re(x)) + mp.im(x) * mp.j
    return mp.radians(x)


class MCalc:
    def __init__(self):
        self.constants = dict(pi=mp.pi, tau=mp.pi * 2, e=mp.e, i=mp.j, _c=299792458)
        self.variables = dict(last=0)
        self.functions0 = dict(rand=mp.rand)
        self.functions1 = dict(abs=mp.fabs, sqrt=mp.sqrt, cbrt=mp.cbrt, log=mp.log10, ln=mp.ln,
                               log2=functools.partial(mp.log, b=2),
                               sin=compose(mp.sin, self._toAngle),
                               cos=compose(mp.cos, self._toAngle),
                               tan=compose(mp.tan, self._toAngle),
                               asin=compose(self._fromAngle, mp.asin),
                               acos=compose(self._fromAngle, mp.acos),
                               atan=compose(self._fromAngle, mp.atan),
                               sinc=compose(mp.sinc, self._toAngle),
                               ceil=mp.ceil, floor=mp.floor, round=mp.nint,
                               frac=mp.frac, sign=mp.sign, re=mp.re, im=mp.im,
                               deg=degrees, rad=radians,
                               )
        self.functions2 = dict(log=mp.log, atan2=compose(self._fromAngle, mp.atan2))
        self.settings = Settings()
        self._lineCounter = -1
        self._lexer = self._createLexer()
        self._parser = self._createParser()

    def calc(self, line):
        try:
            answers = self._parser.parse(self._lexer.lex(line))
            results = []
            for answer in answers:
                if isinstance(answer, Node):
                    try:
                        value = answer.eval()
                        self.variables['last'] = value
                        results.append((True, self.mpfToStr(value)))
                    except ValueError as e:
                        results.append((False, 'Error: ' + str(e)))
                elif isinstance(answer, RuntimeWarning):
                    results.append((False, 'Warning: ' + str(answer)))
                else:
                    results.append((True, answer))
            return results

        except rply.errors.LexingError as e:  # TODO restructure exception handling
            return [(False, self._errorMessage('unexpected character', line, e))]
        except rply.errors.ParsingError as e:
            return [(False, self._errorMessage('parser error', line, e))]
        except ValueError as e:
            return [(False, 'Error: ' + str(e) + '\n')]

    def run(self):
        while True:
            try:
                line = input('> ' if os.isatty(0) else '') + '\n'
                self._lineCounter += 1
                results = self.calc(line)

                for (isResult, result) in results:
                    if isResult:
                        if os.isatty(0):
                            if '=' in result:
                                printIndented(result)
                            elif ':' in result:
                                printIndented(result)
                            elif '.' in result:
                                pos = result.find('.')
                                printIndented('%20s%s' % (result[:pos], result[pos:]))
                            else:
                                printIndented('%20s' % result)
                        else:
                            print(result)
                    else:
                        sys.stderr.write(result)
                        sys.stderr.write('\n')
                        sys.stderr.flush()
            except EOFError:
                break

    def _functionNames(self):
        return list(self.functions0.keys()) + list(self.functions1.keys()) \
               + list(self.functions2.keys())

    def mpfToStr(self, value):
        def toStr(realValue):
            result = mp.nstr(realValue, int(self.settings['digits']))
            return result[0:-2] if result.endswith('.0') else result

        if isinstance(value, mp.mpc):
            real = toStr(mp.re(value))
            imag = toStr(mp.im(value))

            op = '+'

            if imag.startswith("-"):
                imag = imag[1:]
                op = '-'
            if imag == '1':
                imag = ''

            if real == '0':
                if op == '-':
                    return '-%si' % imag
                else:
                    return '%si' % imag
            if imag == '0':
                return real
            return '%s %s %si' % (real, op, imag)
        else:
            return toStr(value)

    def _toAngle(self, x):
        return radians(x) if self.settings['angle'] == 'deg' else x

    def _fromAngle(self, x):
        return degrees(x) if self.settings['angle'] == 'deg' else x

    def _errorMessage(self, message, line, e):
        try:
            line, row, col = MCalc._lineRowColFromIndex(line, e.source_pos.idx)
            errorMessage = ''
            errorMessage += 'Error: %s at %d:%d\n' % (message, self._lineCounter + row, col)
            errorMessage += '  "%s"\n' % line
            errorMessage += '  %s^\n' % ('—' * col)
        except AttributeError:
            errorMessage = 'Error: %s\n' % (message,)
        return errorMessage

    @staticmethod
    def _lineRowColFromIndex(s, index):
        line = ''
        row = 1
        col = 1
        for i in range(len(s)):
            if i < index:
                if s[i] == '\n':
                    line = ''
                    row += 1
                    col = 1
                else:
                    line += s[i]
                    col += 1
            else:
                if s[i] == '\n':
                    break
                line += s[i]
        return line, row, col

    def _createLexer(self):
        lg = rply.LexerGenerator()
        lg.ignore(r'[ \t]+')
        lg.add('HELP', r'help')
        lg.add('COMMAND', r"\.[_a-zA-Z][_a-zA-Z0-9']*")
        lg.add('FUNCTION', r"(%s)[ \t]*\(" % '|'.join(self._functionNames()))
        lg.add('NUMBER', r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
        lg.add('PLUS', r'\+')
        lg.add('MINUS', r'\-')
        lg.add('MULTIPLY', r'\*')
        lg.add('DIVIDE', r'/')
        lg.add('MODULO', r'%')
        lg.add('POWER', r'\^')
        lg.add('FACTORIAL', r'!')
        lg.add('SEMICOLON', r';')
        lg.add('ENTER', r'[\n\r]')
        lg.add('LPAREN', r'\(')
        lg.add('RPAREN', r'\)')
        lg.add('NAME', r"[\w']+")
        lg.add('ASSIGN', r'=')
        lg.add('COMMA', r',')
        lg.add('COLON', r':')
        return lg.build()

    def _createParser(self):
        tokens = ('HELP', 'COMMAND', 'FUNCTION', 'NUMBER', 'PLUS', 'MINUS',
                  'MULTIPLY', 'DIVIDE', 'MODULO', 'POWER', 'FACTORIAL',
                  'SEMICOLON', 'ENTER', 'LPAREN', 'RPAREN', 'NAME',
                  'ASSIGN', 'COMMA', 'COLON')
        precedence = (
            ('left', ('SEMICOLON', 'ENTER')),
            ('right', ('ASSIGN',)),
            ('left', ('PLUS', 'MINUS')),
            ('left', ('MULTIPLY', 'DIVIDE', 'MODULO', 'NUMBER', 'NAME', 'FUNCTION', 'LPAREN')),
            ('right', ('POWER',)),
            ('nonassoc', ('FACTORIAL',)),
        )
        pg = rply.ParserGenerator(tokens, precedence)

        @pg.production('statementList : eos')
        def statementList_empty(_):
            return tuple()

        @pg.production('statementList : statement')
        @pg.production('statementList : statementList eos')
        def statementList_statement(p):
            return p[0]

        @pg.production('statementList : statementList eos statement')
        def statementList_statementList_statement(p):
            return p[0] + p[2]

        @pg.production('statement : command')
        def statement_command(p):
            return p[0]

        @pg.production('statement : setting')
        @pg.production('statement : assignment')
        def statement_novalue(p):
            return p[0]

        @pg.production('statement : signedExpr')
        def statement_expr(p):
            return p[0],

        @pg.production('signedExpr : expr')
        def signedExpr_expr(p):
            return p[0]

        @pg.production('signedExpr : sign expr')
        def signedExpr_sign_expr(p):
            def createUnop(e, s):
                if s == '-':
                    return UnOp(self, 4, e, lambda x: -x, '-')
                else:
                    return UnOp(self, 4, e, lambda x: x, '+')
            return functools.reduce(createUnop, p[0][::-1], p[1])

        @pg.production('command : HELP')
        @pg.production('command : HELP NAME')
        def command_help_name(p):
            subject = p[1].value if len(p) > 1 else 'main'
            if subject in helpTexts:
                return helpTexts[subject] + "\n",
            else:
                return ('No help on "%s" available.' % subject),

        @pg.production('command : COMMAND')
        @pg.production('command : COMMAND NAME')
        def command_command(p):
            command = p[0].value

            if command == '.clear':
                self.variables.clear()
                self.variables['last'] = 0
                return tuple()

            if command == '.del':
                if len(p) == 2:
                    name = p[1].value
                    if name in self.variables:
                        del self.variables[name]
                        if name == 'last':
                            self.variables['last'] = 0
                    return tuple()
                else:
                    return 'Error: .del requires an argument. Type "help .del" for more information',

            if command == '.exit' or command == '.quit':
                raise EOFError

            if command == '.reset':
                self.variables.clear()
                self.variables['last'] = 0
                self.settings.reset()
                return tuple()

            if command == '.vars':
                names = sorted(self.variables.keys())
                if len(names) == 0:
                    return tuple()
                formatString = '%%%ds = %%s' % max(map(len, names))  # use largest name width
                return '\n'.join(map(lambda name: formatString
                                     % (name, self.mpfToStr(self.variables[name])), names)),

            raise ValueError('unknown comand: %s' % command)

        @pg.production('setting : COLON')
        def setting_getall(_):
            formatString = "%%%ds: %%s" % max([len(s) for s in self.settings.keys()])
            return [formatString % (s, self.settings[s]) for s in self.settings.keys()]

        @pg.production('setting : NAME COLON')
        def setting_get(p):
            names = findByPrefix(p[0].value, self.settings.keys())
            if len(names) == 0:
                raise ValueError('unknown setting: %s' % p[0].value)
            return map(lambda s: '%s: %s' % (s, self.settings[s]), names)

        @pg.production('setting : NAME COLON NUMBER')
        @pg.production('setting : NAME COLON NAME')
        def setting_set(p):
            names = findByPrefix(p[0].value, self.settings.keys())
            value = p[2].value
            if len(names) == 1:
                self.settings[names[0]] = value
                return tuple()
            elif len(names) == 0:
                raise ValueError('unknown setting: %s' % p[0].value)
            else:
                raise ValueError('ambiguous setting prefix: %s' % p[0].value)

        @pg.production('assignment : NAME ASSIGN signedExpr')
        def assignment(p):
            name = p[0].value
            self.variables[name] = p[2].eval()
            if name in self._functionNames():
                return RuntimeWarning('there is a function of the same name "%s"' % (name,)),
            return tuple()

        @pg.production('expr : NUMBER')
        def expr_number(p):
            return Number(self, p[0].value)

        @pg.production('expr : NAME')
        def expr_name(p):
            return Name(self, p[0].value)

        @pg.production('sign : PLUS')
        @pg.production('sign : MINUS')
        def sign(p):
            return p[0].value

        @pg.production('sign : sign PLUS')
        @pg.production('sign : sign MINUS')
        def signsign(p):
            return p[0] + p[1].value

        @pg.production('expr : expr PLUS expr')
        def expr_addition(p):
            return BinOp(self, 2, p[0], p[2], mp.fadd, '+')

        @pg.production('expr : expr PLUS sign expr')
        def expr_addition_sign(p):
            return BinOp(self, 2, p[0], signedExpr_sign_expr(p[2:4]), mp.fadd, '+')

        @pg.production('expr : expr MINUS expr')
        def expr_subtraction(p):
            return BinOp(self, 2, p[0], p[2], mp.fsub, '-')

        @pg.production('expr : expr MINUS sign expr')
        def expr_subtraction_sign(p):
            return BinOp(self, 2, p[0], signedExpr_sign_expr(p[2:4]), mp.fsub, '-')

        @pg.production('expr : expr MULTIPLY expr')
        def expr_multiplication(p):
            return BinOp(self, 3, p[0], p[2], mp.fmul, '*')

        @pg.production('expr : expr MULTIPLY sign expr')
        def expr_multiplication_sign(p):
            return BinOp(self, 2, p[0], signedExpr_sign_expr(p[2:4]), mp.fmul, '*')

        @pg.production('expr : expr expr', precedence='MULTIPLY')
        def expr_implicit_multiplication(p):
            # Enables no-parentheses syntax for function names, e.g. "sin 30"
            if isinstance(p[0], Name) and p[0].name not in self.variables \
                    and p[0].name in self.functions1:
                return UnOp(self, 99, p[1], self.functions1[p[0].name], p[0].name, 'function')
            
            # Enables syntax for function names with exponents,
            # e.g. "sin^2 30" or "sin^2(30)
            if isinstance(p[0], BinOp) and p[0].fun == mp.power \
                    and isinstance(p[0].children[0], Name) \
                    and p[0].children[0].name not in self.variables \
                    and p[0].children[0].name in self.functions1:

                name = p[0].children[0].name
                exponent = p[0].children[1]
                funOp = UnOp(self, 99, p[1], self.functions1[name], name, 'function')
                return BinOp(self, 5, funOp, exponent, mp.power, '^', space=False)
                
            return BinOp(self, 3, p[0], p[1], mp.fmul, '*')

        @pg.production('expr : expr DIVIDE expr')
        def expr_division(p):
            return BinOp(self, 3, p[0], p[2], mp.fdiv, '/')

        @pg.production('expr : expr DIVIDE sign expr')
        def expr_division_sign(p):
            return BinOp(self, 2, p[0], signedExpr_sign_expr(p[2:4]), mp.fdiv, '/')

        @pg.production('expr : expr MODULO expr')
        def expr_modulo(p):
            return BinOp(self, 3, p[0], p[2], mp.fmod, '%')

        @pg.production('expr : expr MODULO sign expr')
        def expr_modulo_sign(p):
            return BinOp(self, 2, p[0], signedExpr_sign_expr(p[2:4]), mp.fmod, '%')

        @pg.production('expr : expr POWER expr')
        def expr_power(p):
            return BinOp(self, 5, p[0], p[2], mp.power, '^', space=False)

        @pg.production('expr : expr POWER sign expr')
        def expr_power_sign(p):
            return BinOp(self, 2, p[0], signedExpr_sign_expr(p[2:4]), mp.power, '^', space=False)

        @pg.production('expr : expr FACTORIAL')
        def factexpr_factorial(p):
            return UnOp(self, 6, p[0], mp.factorial, '!', 'postfix')

        @pg.production('expr : LPAREN signedExpr RPAREN')
        def parenexpr_parens(p):
            return p[1]

        @pg.production('expr : function')
        def expr_function(p):
            return p[0]

        @pg.production('function : FUNCTION RPAREN')
        def function0(p):
            name = p[0].value[:-1].strip()  # FUNCTION token includes LPAREN, cut it off
            if name in self.functions0:
                return NoArgsFun(self, self.functions0[name], name)
            else:
                raise ValueError('unknown function: %s' % name)

        @pg.production('function : FUNCTION signedExpr RPAREN')
        def function1(p):
            name = p[0].value[:-1].strip()  # FUNCTION token includes LPAREN, cut it off
            if name in self.functions1:
                return UnOp(self, 99, p[1], self.functions1[name], name, 'function')
            else:
                raise ValueError('unknown function: %s' % name)

        @pg.production('function : FUNCTION signedExpr COMMA signedExpr RPAREN')
        def function2(p):
            name = p[0].value[:-1].strip()  # FUNCTION token includes LPAREN, cut it off
            if name in self.functions2:
                return BinOp(self, 99, p[1], p[3], self.functions2[name], name, 'function')
            else:
                raise ValueError('unknown function: %s' % name)

        @pg.production('eos : ENTER')
        @pg.production('eos : SEMICOLON')
        def endOfStatement(_):
            pass

        p = pg.build()
        for c in p.lr_table.sr_conflicts:
            print(c)
        for c in p.lr_table.rr_conflicts:
            print(c)
        return p


class Settings:
    def __init__(self):
        self._settings = dict()
        self.reset()
        self._validators = dict(angle=Settings.setValidator('deg', 'rad'),
                                precision=Settings.rangeValidator(3),
                                digits=Settings.rangeValidator(1))
        self._sideeffects = dict(digits=self.increasePrecision,
                                 precision=self.setMpmathPrecision)

    def __iter__(self):
        return iter(self._settings.keys())

    def __len__(self):
        return len(self._settings)

    def __getitem__(self, key):
        if key in self._settings:
            return self._settings[key]

    def __setitem__(self, key, value):
        if key not in self._settings:
            raise ValueError('no such setting: %s' % key)
        self._settings[key] = self._validators[key](value)
        if key in self._sideeffects:
            self._sideeffects[key]()

    def reset(self):
        self._settings['angle'] = 'deg'
        self._settings['digits'] = 10
        self._settings['precision'] = 12

    def keys(self):
        return self._settings.keys()

    def increasePrecision(self):
        if self['precision'] < self['digits'] + 2:
            self['precision'] = self['digits'] + 2

    def setMpmathPrecision(self):
        mp.mp.dps = self['precision']
        if self['digits'] + 2 > self['precision']:
            self['digits'] = self['precision'] - 2

    @staticmethod
    def setValidator(*allowedValues):
        def f(x):
            if x in allowedValues:
                return x
            raise ValueError('invalid value: %s. Valid values are: "%s"'
                             % (x, '", "'.join(allowedValues)))
        return f

    @staticmethod
    def rangeValidator(minValue, maxValue = None):
        def f(x):
            try:
                int_x = int(x)
            except ValueError:
                int_x = None

            if int_x is not None and int_x >= minValue and (maxValue is None or int_x <= maxValue):
                return int_x

            message = 'invalid value: %s. Valid values are integral numbers ' % x
            if maxValue is None:
                message += 'larger than or equal to %s' % minValue
            else:
                message += 'between %s and %s (inclusive)' % (minValue, maxValue)
            raise ValueError(message)
        return f


class Node():
    def __init__(self, mcalc, precedence, *children):
        self._mcalc = mcalc
        self.precedence = precedence
        self.children = children

    def __repr__(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def precedence(self):
        return self.precedence

    def reprChild(self, i):
        c = self.children[i]
        if c.precedence() < self.precedence():
            return '(%s)' % repr(c)
        else:
            return repr(c)


class Name(Node):
    def __init__(self, mcalc, name):
        super().__init__(mcalc, 99)
        self.name = name

    def eval(self):
        if self.name in self._mcalc.variables:
            return self._mcalc.variables[self.name]
        if self.name in self._mcalc.constants:
            return self._mcalc.constants[self.name]
        raise ValueError('unknown constant or variable: ' + self.name)

    def __repr__(self):
        return self.name


class Number(Node):
    def __init__(self, mcalc, text):
        super().__init__(mcalc, 99)
        self._value = mp.mpf(text)

    def eval(self):
        return self._value

    def __repr__(self):
        return self._mcalc.mpfToStr(self._value)


class NoArgsFun(Node):
    def __init__(self, mcalc, fun, displayStr):
        super().__init__(mcalc, 99)
        self._fun = fun
        self._displayStr = displayStr

    def eval(self):
        return self._fun()

    def __repr__(self):
        return self._displayStr + '()'


class UnOp(Node):
    def __init__(self, mcalc, precedence, x, fun, displayStr, displayType='prefix', space=False):
        if isinstance(x, str):
            raise RuntimeError()
        super().__init__(mcalc, precedence, x)
        self._fun = fun
        self._displayStr = displayStr
        self._displayType = displayType
        self._space = ' ' if space else ''

    def eval(self):
        return self._fun(self.children[0].eval())

    def __repr__(self):
        if self._displayType == 'prefix':
            return self._displayStr + self._space + self.reprChild(0)
        elif self._displayType == 'postfix':
            return self.reprChild(0) + self._space + self._displayStr
        elif self._displayType == 'function':
            return '%s(%s)' % (self._displayStr, self.reprChild(0))
        else:
            raise ValueError('invalid display type')


class BinOp(Node):
    def __init__(self, mcalc, precedence, x, y, fun, displayStr, displayType='infix', space=True):
        super().__init__(mcalc, precedence, x, y)
        self.fun = fun
        self.displayStr = displayStr
        self.displayType = displayType
        self.space = ' ' if space else ''

    def eval(self):
        return self.fun(self.children[0].eval(), self.children[1].eval())

    def __repr__(self):
        if self.displayType == 'infix':
            return self.reprChild(0) + self.space + self.displayStr + self.space + self.reprChild(1)
        elif self.displayType == 'function':
            return '%s(%s, %s)' % (self.displayStr, repr(self.children[0]), repr(self.children[1]))


def runTests():
    mcalc = MCalc()
    ok = True
    # Test operator precedence
    ok &= _testExpr('1+2*3+4', '11', mcalc)
    ok &= _testExpr('(1+2)*(3+4)', '21', mcalc)
    ok &= _testExpr('4-1', '3', mcalc)
    ok &= _testExpr('2*3^4*5', '810', mcalc)
    ok &= _testExpr('2+3^4+5', '88', mcalc)
    ok &= _testExpr('-2^2', '-4', mcalc)
    ok &= _testExpr('-3!^3!', '-46656', mcalc)
    ok &= _testExpr('5+-1', '4', mcalc)
    ok &= _testExpr('5+-+-1', '6', mcalc)
    ok &= _testExpr('2^3^2', '512', mcalc)
    ok &= _testExpr('1+2sqrt(4)', '5', mcalc)
    ok &= _testExpr('2^4sqrt(4)', '32', mcalc)

    # Test settings
    mcalc.calc('precision:7')  # expect 5 displayed digits
    ok &= _testExpr('pi', '3.1416', mcalc)
    mcalc.calc('digits:10')
    ok &= _testExpr('pi', '3.141592654', mcalc)
    mcalc.calc('angle:deg')
    ok &= _testExpr('sin(30)', '0.5', mcalc)
    mcalc.calc('angle:rad')
    ok &= _testExpr('sin(pi/6)', '0.5', mcalc)

    # Test implicit multiplication
    ok &= _testExpr('-2 3', '-6', mcalc)
    ok &= _testExpr('4(-1)', '-4', mcalc)
    ok &= _testExpr('sqrt=2; 2 sqrt', '4', mcalc)
    ok &= _testExpr('2 sqrt(4)', '4', mcalc)
    ok &= _testExpr('A=7;B=11;C=13; 2A B C', '2002', mcalc)
    ok &= _testExpr('2 5! 3', '720', mcalc)
    ok &= _testExpr('(1+2)(3+4)', '21', mcalc)

    if ok:
        print('All tests passed')


def _testExpr(expr, expectedResult, mcalc=None):
    if mcalc is None:
        mcalc = MCalc()
    results = [x for x in mcalc.calc(expr) if not x[1].startswith('Warning:')]
    if len(results) != 1:
        print('Test failure on "%s": expected one result, but got %d' % (expr, len(results)))
        return False
    isResult, result = results[0]
    if not isResult:
        print('Test failure on "%s": expected a result, but got %s' % (expr, result))
        return False
    if result != expectedResult:
        print('Test failure on "%s": expected %s but got %s' % (expr, expectedResult, result))
        return False
    return True


helpTexts = dict(
main="""
mcalc is an easy to use command line command line calculator with
arbitrary precision. It works with valid mathematical expressions
containing the operators +, -, *, /, %, ^ and ! as well as parentheses ().
Expressions are separated by newline or ;. Use the extended help commands
for more information., and type ".quit" or hit Ctrl-D to quit.


Examples:                           Available extended help commands:
> 2*3^4+(5-1)!; sin(30)
                     186                help commands
                     0.5                help constants
> x = 42; 3x^2; 1 + .1e1                help functions
                    5292                help settings
                       2                help variables
""",
commands="""
The following commands are available:

    .del <variable>     Deletes a variable.
    
    .clear              Deletes all variables.
    
    .exit | .quit       Exits mcalc. The shortcut Ctrl-D also works.
    
    .reset              Deletes all variables and restores the default
                        settings.
    
    .vars               Shows all currently known variables and their
                        values.
""",
constants="""
The following constants are available. You can use them in mathematical
expressions like you would a variable.

    _c                  Speed of light in m/s, 299792458.

    e                   Euler's constant, ~2.718.
    
    i                   Imaginary unit, with i^2 = -1.
    
    pi                  Pi is the ratio of a circle's circumference to its
                        diameter. ~3.142
    
    tau                 2 * Pi. ~6.283

Note that it is possible to create variables with the same name as these
constants. If you do so, the constants will not be accessible any more
until you delete those variables using the .del command.
""",
functions="""
The following functions are available:

    Basic:              abs(x), log(x), log(x, b), ln(x), log2(x),
                        sqrt(x), cbrt(x)

    Trigonometric:      sin(a), cos(a), tan(a), asin(r), acos(r), atan(r),
                        atan2(y, x), sinc(a)
    
    Rounding:           ceil(x), floor(x), round(x)
    
    Conversion:         deg(a), rad(a)
    
    Part Extraction:    frac(x), sign(x), re(c), im(c)
    
    Other:              rand()
""",
settings="""
To display a setting, type its name (including the :). To change a
setting, follow that by the new value, e.g. "angle:rad". The following
settings are available:

    :                   Not a setting itself, but prints all settings.
    
    angle:              Angle mode for trigonometric operations. Valid
                        values are "deg" for degrees or "rad" for radians.
                        
    digits:             The number of digits to display in the results.
                        The precision will be increased if necessary to
                        allow the number of digits. Can be 1 or larger.
                        
    precision:          Determines how many decimal digits are used during
                        calculation. Can be 3 or larger.
""",
variables="""
To assign a value to variable, use the equals sign like so:

> x = 42

You can then use the variable in mathematical expressions, e.g.:

> 3x^2
                    5292

Variable names may contain the letters from a-z in lower- or uppercase,
digits (but not as first character) and the characters _ and '.
To see the list of currently defined variables, use the .vars command.
To delete variables, use the .del command.

The variable "last" will automatically be assigned the result of the last
calculation.        
""")


if mp.libmp.BACKEND != 'gmpy':
    helpTexts['main'] += """
Note: no gmpy module is available. mcalc still works, but it will run much
faster with gmpy or gmpy2 available.
"""


if __name__ == '__main__':
    try:
        if '-t' in sys.argv:
            runTests()
        else:
            calculator = MCalc()
            calculator.run()
    except KeyboardInterrupt:
        sys.exit(-1)