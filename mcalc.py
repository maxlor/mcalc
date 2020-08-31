#!/usr/bin/env python3

# Copyright 2020 Benjamin Lutz
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 3 as published by the
# Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.

import argparse
import appdirs
import functools
import mpmath as mp
import os
try:
    import readline  # intentional, even if an IDE claims it's unused
except ImportError:  # may be unavailable on Windows
    pass
import rply
import sys


_progName = 'mcalc'
_version = '1.0.2'
_rcFile = os.path.join(appdirs.user_config_dir(_progName, False), _progName + '.rc')


def compose(f, g):
    return lambda *x: f(g(*x))


def _printIndented(*things):
    for thing in things:
        for line in str(thing).splitlines():
            print('    %s' % line)


def _findByPrefix(prefix, strings):
    return [s for s in strings if s.startswith(prefix)]


def _deg(x):
    if isinstance(x, mp.mpc):
        return mp.degrees(mp.re(x)) + mp.im(x) * mp.j
    return mp.degrees(x)


def _rad(x):
    if isinstance(x, mp.mpc):
        return mp.radians(mp.re(x)) + mp.im(x) * mp.j
    return mp.radians(x)


class MCalc:
    def __init__(self):
        self.scopedVariableStack = []
        self.constants = dict(pi=mp.pi, π=mp.pi, tau=mp.pi * 2, τ=mp.pi * 2, e=mp.e, i=mp.j, _c=299792458)
        self.variables = dict(last=0)
        self.functionParameters = [dict()]  # stack of dictionaries with guard
        self.functions = {
            0: dict(rand=mp.rand),
            1: dict(abs=mp.fabs, sqrt=mp.sqrt, cbrt=mp.cbrt, log=mp.log10, ln=mp.ln,
                    log2=functools.partial(mp.log, b=2), sin=compose(mp.sin, self._toAngle),
                    cos=compose(mp.cos, self._toAngle), tan=compose(mp.tan, self._toAngle),
                    asin=compose(self._fromAngle, mp.asin),
                    acos=compose(self._fromAngle, mp.acos),
                    atan=compose(self._fromAngle, mp.atan),
                    sinh=mp.sinh, cosh=mp.cosh, tanh=mp.tanh,
                    asinh=mp.asinh, acosh=mp.acosh, atanh=mp.atanh,
                    sinc=compose(mp.sinc, self._toAngle), gamma=mp.gamma, binom=mp.binomial,
                    ceil=mp.ceil, floor=mp.floor, round=mp.nint, frac=mp.frac, sign=mp.sign,
                    re=mp.re, im=mp.im, conj=mp.conj, deg=_deg, rad=_rad),
            2: dict(log=mp.log, atan2=compose(self._fromAngle, mp.atan2))
        }
        self.userFunctions = dict()
        self.settings = _Settings()
        self.substitutions = {'¹': '^(1)', '²': '^(2)', '³': '^(3)', '⁴': '^(4)', '⁵': '^(5)',
                              '⁶': '^(6)', '⁷': '^(7)', '⁸': '^(8)', '⁹': '^(9)', 'ⁱ': '^(i)',
                              '½': '(1/2)', '⅓': '(1/3)', '⅔': '(2/3)', '¼': '(1/4)',
                              '¾': '(3/4)', '⅕': '(1/5)', '⅖': '(2/5)', '⅗': '(3/5)',
                              '⅘': '(4/5)', '⅐': '(1/7)', '⅑': '(1/9)', '⅒': '(1/10)'}
        self._inputFile = None
        self._lineCounter = -1
        self._lexer = self._createLexer()
        self._parser = self._createParser()

    def calc(self, line):
        """
        Run a line of text through the calculator. Line can actually contain
        several lines, or several statements separated by semicolon.

        The method returns a list of results and warnings as tuples of type
        (bool, str), where the first boolean part is True if the second part
        is a result, and False if it is a warning or error.

        The ordering of the results will be the same as the ordering of
        input statements; but if a statement doesn't produce a result or
        warning, nothing will be added to the results list, so the number of
        results may be less than the number of statements.

        On error, processing of the line is aborted; if the line contains
        several statements, statements following the error will not be
        processed.

        :param line: a string containing calculator input
        :return:     a list of tuples with results, warnings and errors
        """
        for key, value in self.substitutions.items():
            line = line.replace(key, value)

        try:
            parsedObjects = self._parser.parse(self._lexer.lex(line))
            results = []
            for parsedObj in parsedObjects:
                if isinstance(parsedObj, _AbstractExpr):
                    parsedObj.fixup('functionCalls')
                    try:
                        value = parsedObj.eval()
                        self.variables['last'] = value
                        results.append((True, self._mpfToStr(value)))
                    except ValueError as e:
                        results.append((False, 'Error: ' + str(e)))
                    except ZeroDivisionError:
                        results.append((False, 'Error: division by zero'))
                elif isinstance(parsedObj, str):
                    results.append((True, parsedObj))
                elif isinstance(parsedObj, RuntimeWarning):
                    results.append((False, 'Warning: ' + str(parsedObj)))
                elif isinstance(parsedObj, RuntimeError):
                    results.append((False, 'Error: ' + str(parsedObj)))
                elif isinstance(parsedObj, tuple) and len(parsedObj) == 0:
                    pass
                else:
                    print("parsedObj (%s) = %s" % (type(parsedObj), parsedObj))
                    assert False
            return results

        except rply.errors.LexingError as e:
            return [(False, self._errorMessage('unexpected character', line, e))]
        except rply.errors.ParsingError as e:
            return [(False, self._errorMessage('parser error', line, e))]
        except ValueError as e:
            return [(False, 'Error: ' + str(e) + '\n')]

    def run(self, skipRcFile):
        """
        Run the calculator. If a TTY is detected, it will show a prompt to the
        user and allow command line editing features (courtesy of readline).

        If no TTY is detected, it'll read from STDIN until EOF, process the
        data and then exit.

        :param skipRcFile: whether to run the RC file at startup
        """
        def showResults(results):
            for (isResult, result) in results:
                assert isinstance(result, str)
                if isResult:
                    if os.isatty(0):
                        if '=' in result:
                            _printIndented(result)
                        elif ':' in result:
                            _printIndented(result)
                        elif '.' in result:
                            pos = result.find('.')
                            _printIndented('%20s%s' % (result[:pos], result[pos:]))
                        else:
                            _printIndented('%20s' % result)
                    else:
                        print(result)
                else:
                    sys.stderr.write(result)
                    sys.stderr.write('\n')
                    sys.stderr.flush()

        if not skipRcFile:
            showResults(self._runRcFile())

        while True:
            try:
                line = input('> ' if os.isatty(0) else '') + '\n'
                self._lineCounter += 1
                try:
                    showResults(self.calc(line))
                except KeyboardInterrupt:
                    _printIndented("%20s" % "interrupted")

            except EOFError:
                break

    def _runRcFile(self):
        """Run the RC file."""
        result = []
        if os.path.isfile(_rcFile):
            with open(_rcFile) as f:
                self._inputFile = _rcFile
                try:
                    for line in f.readlines():
                        self._lineCounter += 1
                        result += self.calc(line)
                finally:
                    self._inputFile = None
                    self._lineCounter = -1
        return result

    def _haveFunction(self, name, checkUserFunctions=True, paramCount=None):
        """
        Check whether the function called "name" exists.

        If "parameters" is 0 or larger, the function must have exiactly that many
        parameters to match, otherwise any number of parameters will do. If it is
        negative, it the function must have abs(parameters) or more to match. If
        "parameters" is None, any number of parameters will do.

        :param name:               the function name to search for
        :param checkUserFunctions: whether to check user functions for a match too
        :param paramCount:         the number of parameters
        :return:                   True if a function is found, False otherwise
        """
        def lookIn(paramsFuncDict):
            if paramCount is None:
                for funcDict in paramsFuncDict.values():
                    if name in funcDict:
                        return True
            elif paramCount < 0:
                for noParams, funDict in paramsFuncDict.items():
                    if noParams < -paramCount:
                        continue
                    if name in funDict:
                        return True
            elif paramCount in paramsFuncDict and name in paramsFuncDict[paramCount]:
                return True
            return False

        if lookIn(self.functions):
            return True
        if checkUserFunctions and lookIn(self.userFunctions):
            return True
        return False

    def _mpfToStr(self, value):
        """
        Render an mpmath mpf or mpc value as string, taking into account the
        "digits" setting.

        :param value: the number
        :return:      a string representation of the number
        """
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
        """Convert x from the current angle setting to radians."""
        return _rad(x) if self.settings['angle'] == 'deg' else x

    def _fromAngle(self, x):
        """Convert x from radians to the current angle setting."""
        return _deg(x) if self.settings['angle'] == 'deg' else x

    def _errorMessage(self, message, line, e):
        """Creates a nicely formatted error message."""
        try:
            line, row, col = MCalc._lineRowColFromIndex(line, e.source_pos.idx)
            errorMessage = ''
            if self._inputFile is not None:
                errorMessage += 'Error: %s at %s:%d:%d\n' % (message, self._inputFile, self._lineCounter + row, col)
            else:
                errorMessage += 'Error: %s at %d:%d\n' % (message, self._lineCounter + row, col)
            errorMessage += '  "%s"\n' % line
            errorMessage += '  %s^\n' % ('—' * col)
        except AttributeError:
            errorMessage = 'Error: %s\n' % (message,)
        return errorMessage

    @staticmethod
    def _lineRowColFromIndex(s, index):
        """Convert an index into line, row, column."""
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

    @staticmethod
    def _createLexer():
        """Create the lexer."""
        lg = rply.LexerGenerator()
        lg.ignore(r'[ \t]+')
        lg.add('HELP', r'help')
        lg.add('COMMAND', r"\.[_a-zA-Z][_a-zA-Z0-9']*")
        lg.add('NUMBER', r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
        lg.add('PLUS', r'\+')
        lg.add('MINUS', r'[-−]')
        lg.add('POWER', r'(?:\^|\*\*)')
        lg.add('MULTIPLY', r'[*⋅]')
        lg.add('DIVIDE', r'[/÷]')
        lg.add('MODULO', r'%')
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
        """Create the parser."""
        tokens = ('HELP', 'COMMAND', 'NUMBER', 'PLUS', 'MINUS',
                  'MULTIPLY', 'DIVIDE', 'MODULO', 'POWER', 'FACTORIAL',
                  'SEMICOLON', 'ENTER', 'LPAREN', 'RPAREN', 'NAME',
                  'ASSIGN', 'COMMA', 'COLON')
        precedence = (
            ('left', ('SEMICOLON', 'ENTER')),
            ('right', ('ASSIGN',)),
            ('left', ('COMMA', 'RPAREN')),
            ('left', ('PLUS', 'MINUS')),
            ('left', ('MULTIPLY', 'DIVIDE', 'MODULO', 'NUMBER', 'NAME', 'LPAREN')),
            ('right', ('POWER',)),
            ('nonassoc', ('FACTORIAL',)),
            ('right', ('SIGNSIGN',)),
        )

        def maybe_name(t):
            if isinstance(t, rply.Token) and t.name == 'NAME':
                return Name(self, t.value)
            else:
                return t

        def signed(sign, x):
            def identity(x):
                return x

            fun = mp.fneg if sign == '-' else identity
            return UnOp(self, 4, x, fun, sign)

        pg = rply.ParserGenerator(tokens, precedence, cache_id='mcalc')

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

        @pg.production('statement : sum')
        def statement_sum(p):
            return ExprRoot(self, p[0]),

        @pg.production('command : HELP')
        @pg.production('command : HELP NAME')
        def command_help_name(p):
            subject = p[1].value if len(p) > 1 else 'main'
            if subject in _helpTexts:
                return _helpTexts[subject] + "\n",
            else:
                return 'No help on "%s" available.' % subject,

        @pg.production('command : COMMAND')
        @pg.production('command : COMMAND sum')
        def command_command(p):
            command = p[0].value

            if command == '.clear':
                self.variables.clear()
                self.variables['last'] = 0
                return tuple()

            if command == '.del':
                if len(p) != 2:
                    return 'Error: .del requires an argument. Type "help .del" for more information',

                if isinstance(p[1], rply.Token) and p[1].name == 'NAME':
                    name = p[1].value
                    if name not in self.variables:
                        return RuntimeWarning('no such variable'),

                    del self.variables[name]
                    if name == 'last':
                        self.variables['last'] = 0
                    return tuple()
                elif isinstance(p[1], FunCall) or isinstance(p[1], _AbstractExpr):
                    expr = ExprRoot(self, p[1])
                    expr.fixup('functionCalls')
                    name = repr(expr)
                    argCount = None

                    if '()' in name:
                        argCount = 0
                    elif '(' in name:
                        argCount = name.count(',') + 1
                    if argCount is None:
                        return RuntimeWarning('no such function'),

                    funName = name[:name.find('(')]
                    if funName in self.userFunctions[argCount]:
                        del self.userFunctions[argCount][funName]
                        return tuple()

                return RuntimeWarning('No such variable or function')

            if command == '.exit' or command == '.quit':
                raise EOFError

            if command == '.reset':
                self.variables.clear()
                self.variables['last'] = 0
                self.settings.reset()
                return tuple()

            if command == '.vars':
                variables = ''
                if len(self.variables) > 0:
                    s = '%%%ds = %%s' % max(map(len, self.variables.keys()))  # use largest name width
                    variables = '\n'.join([s % (name, self._mpfToStr(self.variables[name]))
                                           for name in sorted(self.variables)])

                functions = ''
                funDict = dict()
                for argCount in self.userFunctions:
                    for name, funDef in self.userFunctions[argCount].items():
                        fullname = '%s(%s)' % (name, ', '.join([x for x in funDef[0]]))
                        funDict[fullname] = funDef[1]
                if len(funDict) > 0:
                    s = '%%%ds = %%s' % max(map(len, funDict.keys()))
                    functions = '\n'.join([s % (name, funDict[name]) for name in sorted(funDict)])

                if variables != '' and functions != '':
                    return '%s\n\n%s' % (variables, functions),
                return variables + functions,

            raise ValueError('unknown comand: %s' % command)

        @pg.production('setting : COLON')
        def setting_getall(_):
            formatString = "%%%ds: %%s" % max([len(s) for s in self.settings.keys()])
            return [formatString % (s, self.settings[s]) for s in self.settings.keys()]

        @pg.production('setting : NAME COLON')
        def setting_get(p):
            names = _findByPrefix(p[0].value, self.settings.keys())
            if len(names) == 0:
                raise ValueError('unknown setting: %s' % p[0].value)
            return map(lambda s: '%s: %s' % (s, self.settings[s]), names)

        @pg.production('setting : NAME COLON NUMBER')
        @pg.production('setting : NAME COLON NAME')
        def setting_set(p):
            names = _findByPrefix(p[0].value, self.settings.keys())
            value = p[2].value
            if len(names) == 1:
                self.settings[names[0]] = value
                return tuple()
            elif len(names) == 0:
                raise ValueError('unknown setting: %s' % p[0].value)
            else:
                raise ValueError('ambiguous setting prefix: %s' % p[0].value)

        @pg.production('assignment : NAME ASSIGN sum')
        def assignment_name(p):
            name = p[0].value
            try:
                self.variables[name] = p[2].eval()
            except ValueError as e:
                return RuntimeError(str(e)),
            except ZeroDivisionError:
                return RuntimeError('division by zero'),

            if self._haveFunction(name):
                return RuntimeWarning('there is a function of the same name "%s"' % (name,)),
            return tuple()

        @pg.production('assignment : function ASSIGN sum')
        def assignment_function_noParams(p):
            if 0 not in self.userFunctions:
                self.userFunctions[0] = dict()
            name = p[0].name
            self.userFunctions[0][name] = (tuple(), p[2])
            if name in self.variables:
                return RuntimeWarning('there is a variable of the same name "%s"' % (name,)),
            if self._haveFunction(name, False, 0):
                return RuntimeWarning('there is a built-in function of the same name "%s"' % (name,)),
            return tuple()

        @pg.production('assignment : sum ASSIGN sum')
        def assignment_function(p):
            assert isinstance(p[0], BinOp) and p[0].fun == mp.fmul
            assert isinstance(p[0].children[0], Name)
            name = p[0].children[0].name
            if isinstance(p[0].children[1], Name):
                arguments = p[0].children[1].name,
            elif isinstance(p[0].children[1], ArgListNode):
                arguments = tuple([a.name for a in p[0].children[1].children])
            else:
                assert False

            if len(arguments) not in self.userFunctions:
                self.userFunctions[len(arguments)] = dict()
            exprRoot = ExprRoot(self, p[2])
            exprRoot.fixup('functionCalls')
            self.userFunctions[len(arguments)][name] = (arguments, exprRoot)
            return tuple()

        @pg.production('sum : product')
        @pg.production('sum : NAME')
        def sum_product(p):
            return maybe_name(p[0])

        @pg.production('sum : sign product')
        @pg.production('sum : sign NAME')
        def sum_product(p):
            return signed(p[0], maybe_name(p[1]))

        @pg.production('sum : sum sign product')
        @pg.production('sum : sum sign NAME')
        @pg.production('sum : NAME sign product')
        @pg.production('sum : NAME sign NAME')
        def sum_sum_sign_product(p):
            fun = mp.fsub if p[1] == '-' else mp.fadd
            return BinOp(self, 2, maybe_name(p[0]), maybe_name(p[2]), fun, p[1])

        @pg.production('product : operand')
        def product_operand(p):
            return p[0]

        @pg.production('product : product MULTIPLY product')
        @pg.production('product : product MULTIPLY NAME')
        @pg.production('product : NAME MULTIPLY product')
        @pg.production('product : NAME MULTIPLY NAME')
        def product_product_multiply_product(p):
            return BinOp(self, 3, maybe_name(p[0]), maybe_name(p[2]), mp.fmul, '*')

        @pg.production('product : product DIVIDE product')
        @pg.production('product : product DIVIDE NAME')
        @pg.production('product : NAME DIVIDE product')
        @pg.production('product : NAME DIVIDE NAME')
        def product_product_divide_product(p):
            return BinOp(self, 3, maybe_name(p[0]), maybe_name(p[2]), mp.fdiv, '/')

        @pg.production('product : product MODULO product')
        @pg.production('product : product MODULO NAME')
        @pg.production('product : NAME MODULO product')
        @pg.production('product : NAME MODULO NAME')
        def product_product_modulo_product(p):
            return BinOp(self, 3, maybe_name(p[0]), maybe_name(p[2]), mp.fmod, '%')

        @pg.production('product : product MULTIPLY sign operand')
        @pg.production('product : product MULTIPLY sign NAME')
        @pg.production('product : NAME MULTIPLY sign operand')
        @pg.production('product : NAME MULTIPLY sign NAME')
        def product_product_multiply_sign_operand(p):
            return BinOp(self, 3, maybe_name(p[0]), signed(p[1], maybe_name(p[2])), mp.fmul, '*')

        @pg.production('product : product DIVIDE sign operand')
        @pg.production('product : product DIVIDE sign NAME')
        @pg.production('product : NAME DIVIDE sign operand')
        @pg.production('product : NAME DIVIDE sign NAME')
        def product_product_divide_sign_operand(p):
            return BinOp(self, 3, maybe_name(p[0]), signed(p[1], maybe_name(p[2])), mp.fdiv, '/')

        @pg.production('product : product MODULO sign operand')
        @pg.production('product : product MODULO sign NAME')
        @pg.production('product : NAME MODULO sign operand')
        @pg.production('product : NAME MODULO sign NAME')
        def modulo_sign_operand(p):
            return BinOp(self, 3, maybe_name(p[0]), signed(p[1], maybe_name(p[2])), mp.fmod, '%')

        @pg.production('product : product operand')
        @pg.production('product : product NAME')
        @pg.production('product : NAME operand')
        @pg.production('product : NAME NAME')
        def product_product_operand(p):
            return BinOp(self, 3, maybe_name(p[0]), maybe_name(p[1]), mp.fmul, '*', implicit=True)

        @pg.production('operand : operand POWER operand')
        @pg.production('operand : operand POWER NAME')
        @pg.production('operand : NAME POWER operand')
        @pg.production('operand : NAME POWER NAME')
        def operand_operand_power_operand(p):
            return BinOp(self, 5, maybe_name(p[0]), maybe_name(p[2]),
                         mp.power, '^', space=False)

        @pg.production('operand : operand POWER sign operand')
        @pg.production('operand : operand POWER sign NAME')
        @pg.production('operand : NAME POWER sign operand')
        @pg.production('operand : NAME POWER sign NAME')
        def operand_operand_power_sign_operand(p):
            return BinOp(self, 5, maybe_name(p[0]), signed(p[1], maybe_name(p[2])),
                         mp.power, '^', space=False)

        @pg.production('operand : operand FACTORIAL')
        @pg.production('operand : NAME FACTORIAL')
        def operand_operand_factorial(p):
            return UnOp(self, 6, maybe_name(p[0]), mp.factorial, '!', 'postfix')

        @pg.production('operand : NUMBER')
        def operand_number(p):
            return Number(self, p[0].value)

        @pg.production('operand : LPAREN sum RPAREN')
        def operand_lparen_sum_rparen(p):
            return p[1]

        @pg.production('operand : function')
        def operand_function(p):
            return p[0]

        @pg.production('sign : PLUS')
        @pg.production('sign : MINUS')
        def sign_plus(p):
            return p[0].value

        @pg.production('sign : sign sign', precedence='SIGNSIGN')
        def sign_sign_sign(p):
            if p[0] == '+':
                return p[1]
            else:
                return '+' if p[1] == '-' else '-'

        @pg.production('function : NAME LPAREN RPAREN')
        def function(p):
            return FunCall(self, p[0].value)

        @pg.production('operand : LPAREN argumentList RPAREN')
        def operand_argumentList(p):
            return ArgListNode(self, *p[1])

        @pg.production('argumentList : sum argumentListContinuation')
        @pg.production('argumentList : NAME argumentListContinuation')
        def argumentList_sum(p):
            return (maybe_name(p[0]),) + p[1]

        @pg.production('argumentListContinuation : COMMA sum')
        def argumentListContinuation_sum(p):
            return maybe_name(p[1]),

        @pg.production('argumentListContinuation : COMMA sum argumentListContinuation')
        def argumentListContinuation_sum_argumentListContinuation(p):
            return (maybe_name(p[1]),) + p[2]

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


class _Settings:
    """Settings helper class used by MCalc. Includes setter validation."""
    def __init__(self):
        self._settings = dict(angle='deg', digits=10, precision=mp.mp.dps)
        self._validators = dict(angle=_Settings.setValidator('deg', 'rad'),
                                precision=_Settings.rangeValidator(3),
                                digits=_Settings.rangeValidator(1))
        self._sideeffects = dict(digits=self.increasePrecision,
                                 precision=self.setMpmathPrecision)
        self.reset()

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
        self['angle'] = 'deg'
        self['digits'] = 10
        self['precision'] = 15

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
    def rangeValidator(minValue, maxValue=None):
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


class _AbstractExpr:
    """ Abstract base class for expression tree nodes."""
    def __init__(self, mcalc, precedence, *children):
        """
        Initialize the node.

        :param mcalc:      reference to an MCalc object
        :param precedence: the precedence of this node; 99 is used if there is no precedence
        :param children:   the child nodes
        """
        self.mcalc = mcalc
        self.precedence = precedence
        self.parent = None
        self.children = list(children)
        for child in self.children:
            child.parent = self

    def __repr__(self):
        raise NotImplementedError

    def eval(self):
        """Evaluate this note and return an mpf or mpc number."""
        raise NotImplementedError

    def fixup(self, what=None):
        """
        Fix up the syntax tree after parsing.

        Only one value for what is currently implemented:

        functionCalls
          Turn multiplications into function calls where appropriate. After
          parsing "sin 45", we have a multiplication of "sin" and "45", so
          we detect where the name refers to a function instead of a variable
          and replace the multiplication by a function call.
        """
        def hasExprAncestor(node):
            while not isinstance(node, ExprRoot):
                node = node.parent
                if node is None:
                    return False
            return True

        # Iterate through child nodes, but stop if this node has been deleted
        # by the child. It's the child's responsibility to resume the fixup.
        for i in range(len(self.children)):
            self.children[i].fixup(what)
            if not hasExprAncestor(self):
                return  # This node has been deleted in fixup

    def replaceChild(self, oldChild, newChild):
        """Replace the child oldChild with newChild."""
        for i, c in enumerate(self.children):
            if c is oldChild:
                oldChild.parent = None
                newChild.parent = self
                self.children[i] = newChild
                return oldChild
        raise ValueError('child not found')

    def reprChild(self, i):
        """
        Return a string representation of the i-th child, with parentheses
        added if required by operator precedence rules.
        """
        c = self.children[i]
        if c.precedence < self.precedence:
            return '(%s)' % repr(c)
        else:
            return repr(c)


class ExprRoot(_AbstractExpr):
    """A guard node that serves as the root of an expression tree."""
    def __init__(self, mcalc, expr):
        super().__init__(mcalc, 99, expr)

    def __repr__(self):
        return repr(self.children[0])

    def eval(self):
        return self.children[0].eval()


class Name(_AbstractExpr):
    """A node that contains a name, typically referring to a variable or constant."""
    def __init__(self, mcalc, name):
        super().__init__(mcalc, 99)
        self.name = name

    def eval(self):
        if self.name in self.mcalc.functionParameters[-1]:
            return self.mcalc.functionParameters[-1][self.name]
        if self.name in self.mcalc.variables:
            return self.mcalc.variables[self.name]
        if self.name in self.mcalc.constants:
            return self.mcalc.constants[self.name]
        raise ValueError('unknown constant or variable: ' + self.name)

    def __repr__(self):
        return self.name

    def fixup(self, what=None):
        if what == 'functionCalls':
            pass

            isFunction = (self.name not in self.mcalc.variables and
                          self.mcalc._haveFunction(self.name, True, -1))

            if isFunction:
                def isLeftSideOfImplicitMultiplication(node):
                    return (node.parent is not None and isinstance(node.parent, BinOp) and
                            node.parent.fun == mp.fmul and node.parent.implicit and
                            node.parent.children[0] == node)

                node = self

                # Go up through the AST, until we arrive at a left child of an
                # implicit multiplication
                while not isLeftSideOfImplicitMultiplication(node):
                    if node.parent is None:
                        raise ValueError('Syntax error: cannot find argument for function "%s"' % self.name)
                    node = node.parent

                multiplication = node.parent
                argument = multiplication.children[1]
                if isinstance(argument, ArgListNode):
                    funNode = FunCall(self.mcalc, self.name, *argument.children)
                else:
                    funNode = FunCall(self.mcalc, self.name, argument)
                root = node.parent.parent
                root.replaceChild(node.parent, node)
                self.parent.replaceChild(self, funNode)
                funNode.fixup(what)


class Number(_AbstractExpr):
    """A node containing a number."""
    def __init__(self, mcalc, text):
        super().__init__(mcalc, 99)
        self._value = mp.mpf(text)

    def eval(self):
        return self._value

    def __repr__(self):
        return self.mcalc._mpfToStr(self._value)


class UnOp(_AbstractExpr):
    """A node containing an unary operator."""
    def __init__(self, mcalc, precedence, x, fun, displayStr, displayType='prefix', space=False):
        """
        Initialize the UnOp.

        :param mcalc:       reference to an MCalc object
        :param precedence:  the precedence of this node (int between 1 and 99)
        :param x:           the operator argument, an expression
        :param fun:         the function to apply to x when evaluating this node
        :param displayStr:  a string representation of the operator
        :param displayType: how to render this operator, "prefix" or "postfix"
        :param space:       whether to but a space between the operator and x when rendering
        """
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


class BinOp(_AbstractExpr):
    """A node containing an binary operator."""
    def __init__(self, mcalc, precedence, x, y, fun, displayStr, displayType='infix', space=True,
                 implicit=False):
        """
        Initialize the BinOp.

        :param mcalc:       reference to an MCalc object
        :param precedence:  the precedence of this node
        :param x:           the first operator argument
        :param y:           the second operator argument
        :param fun:         the function to apply to x, y when evaluating this node
        :param displayStr:  a string representation of the operator
        :param displayType: how to render this operator, "infix" or "function"
        :param space:       whether to but a space between the operator and x when rendering
        :param implicit:    whether this operator is implicit
        """
        super().__init__(mcalc, precedence, x, y)
        self.fun = fun
        self.displayStr = displayStr
        self.displayType = displayType
        self.space = space
        self.implicit = implicit

    def eval(self):
        return self.fun(self.children[0].eval(), self.children[1].eval())

    def __repr__(self):
        space = ' ' if self.space else ''
        if self.displayType == 'infix':
            return self.reprChild(0) + space + self.displayStr + space + self.reprChild(1)
        elif self.displayType == 'function':
            return '%s(%s, %s)' % (self.displayStr, repr(self.children[0]), repr(self.children[1]))


class FunCall(_AbstractExpr):
    """A node containing a function call."""
    def __init__(self, mcalc, name, *arguments):
        super().__init__(mcalc, 99, *arguments)
        self.name = name

    def eval(self):
        paramCount = len(self.children)
        parameters = [child.eval() for child in self.children]
        m = self.mcalc

        if paramCount in m.userFunctions and self.name in m.userFunctions[paramCount]:
            paramNames, expr = m.userFunctions[paramCount][self.name]
            assert len(paramNames) == len(self.children)
            paramDict = dict()
            for i, paramName in enumerate(paramNames):
                paramDict[paramName] = parameters[i]
            self.mcalc.functionParameters.append(paramDict)
            result = expr.eval()
            del self.mcalc.functionParameters[-1]
            return result
        if paramCount in m.functions and self.name in m.functions[paramCount]:
            return m.functions[paramCount][self.name](*parameters)
        else:
            raise ValueError('unknown function: "%s"' % (self.name,))

    def __repr__(self):
        return '%s(%s)' % (self.name, ', '.join([repr(child) for child in self.children]))


class ArgListNode(_AbstractExpr):
    """An argument list for a function call."""
    def __init__(self, mcalc, *arguments):
        super().__init__(mcalc, 99, *arguments)

    def eval(self):
        raise RuntimeError("syntax error: argument list without function")

    def __repr__(self):
        return '(%s)' % ', '.join([repr(child) for child in self.children])


def runTests():
    """Run unit tests."""
    mcalc = MCalc()
    ok = True
    ok &= _testExpr('1', '1', mcalc)
    ok &= _testExpr('-1', '-1', mcalc)
    ok &= _testExpr('--1', '1', mcalc)

    # Test operators
    ok &= _testExpr('1+2', '3', mcalc)
    ok &= _testExpr('1+-2', '-1', mcalc)
    ok &= _testExpr('-1+2', '1', mcalc)
    ok &= _testExpr('-1+-2', '-3', mcalc)
    ok &= _testExpr('1-2', '-1', mcalc)
    ok &= _testExpr('2*3', '6', mcalc)
    ok &= _testExpr('2⋅3', '6', mcalc)
    ok &= _testExpr('2 3', '6', mcalc)
    ok &= _testExpr('6/5', '1.2', mcalc)
    ok &= _testExpr('6÷5', '1.2', mcalc)
    ok &= _testExpr('7%5', '2', mcalc)
    ok &= _testExpr('2^9', '512', mcalc)
    ok &= _testExpr('2**9', '512', mcalc)
    ok &= _testExpr('6!', '720', mcalc)

    # Test operator precedence
    ok &= _testExpr('-2 3', '-6', mcalc)
    ok &= _testExpr('1+2*3+4', '11', mcalc)
    ok &= _testExpr('1+2 3+4', '11', mcalc)
    ok &= _testExpr('2 3+4 5', '26', mcalc)
    ok &= _testExpr('(1+2)*(3+4)', '21', mcalc)
    ok &= _testExpr('4-1', '3', mcalc)
    ok &= _testExpr('2*3^4*5', '810', mcalc)
    ok &= _testExpr('2 3^4 5', '810', mcalc)
    ok &= _testExpr('2+3^4+5', '88', mcalc)
    ok &= _testExpr('-2^2', '-4', mcalc)
    ok &= _testExpr('-3!^3!', '-46656', mcalc)
    ok &= _testExpr('5+-1', '4', mcalc)
    ok &= _testExpr('5+-+-1', '6', mcalc)
    ok &= _testExpr('2^3^2', '512', mcalc)
    ok &= _testExpr('sqrt 4', '2', mcalc)
    ok &= _testExpr('sqrt(4)', '2', mcalc)
    ok &= _testExpr('log(256, 2)', '8', mcalc)
    ok &= _testExpr('1+2sqrt(4)', '5', mcalc)
    ok &= _testExpr('2^4sqrt(4)', '32', mcalc)
    ok &= _testExpr('3^sqrt(4)', '9', mcalc)
    ok &= _testExpr('sqrt(4)2', '4', mcalc)
    ok &= _testExpr('2 * 3 * sqrt 4', '12', mcalc)
    ok &= _testExpr('sqrt sqrt 16', '2', mcalc)
    ok &= _testExpr('sqrt^2(1+1)', '2', mcalc)
    ok &= _testExpr('sqrt² 2', '2', mcalc)
    ok &= _testExpr('2¹', '2', mcalc)
    ok &= _testExpr('2²', '4', mcalc)
    ok &= _testExpr('2³', '8', mcalc)
    ok &= _testExpr('2³²', '512', mcalc)
    ok &= _testExpr('1/0', 'Error: division by zero', mcalc, expectError=True)
    ok &= _testExpr('a=1/0', 'Error: division by zero', mcalc, expectError=True)

    # Test settings
    mcalc.calc('precision:7')  # expect 5 displayed digits
    ok &= _testExpr('pi', '3.1416', mcalc)
    mcalc.calc('digits:10')
    ok &= _testExpr('pi', '3.141592654', mcalc)
    mcalc.calc('angle:deg')
    ok &= _testExpr('sin(30)', '0.5', mcalc)
    ok &= _testExpr('atan2(1, 1)', '45', mcalc)
    mcalc.calc('angle:rad')
    ok &= _testExpr('sin(pi/6)', '0.5', mcalc)

    # Test implicit multiplication
    ok &= _testExpr('-2 3', '-6', mcalc)
    ok &= _testExpr('4(-1)', '-4', mcalc)
    ok &= _testExpr('sqrt=10; 2 sqrt', '20', mcalc)
    ok &= _testExpr('sqrt 4', '40', mcalc)
    ok &= _testExpr('sqrt (4)', '40', mcalc)
    ok &= _testExpr('2 sqrt 4', '80', mcalc)
    ok &= _testExpr('2 sqrt (4)', '80', mcalc)
    ok &= _testExpr('A=7;B=11;C=13; 2A B C', '2002', mcalc)
    ok &= _testExpr('2 5! 3', '720', mcalc)
    ok &= _testExpr('(1+2)(3+4)', '21', mcalc)

    # Test user functions
    ok &= _testExpr('.reset;f0()=1.2; f0()', '1.2', mcalc)
    ok &= _testExpr('f1 x=1+x; f1(2)', '3', mcalc)
    ok &= _testExpr('f1 2', '3', mcalc)
    ok &= _testExpr('f1(f1(2))', '4', mcalc)
    ok &= _testExpr('f1a(x)=10+x; f1a(2)', '12', mcalc)
    ok &= _testExpr('f2(a, b)=sqrt(a^2+b^2); f2(3, 4)', '5', mcalc)
    ok &= _testExpr('f3(a, b, c)=a+b+c; f3(1, 2, 3)', '6', mcalc)
    ok &= _testExpr('id x = x; id 1', '1', mcalc)
    ok &= _testExpr('rand() = 0.5; rand()', '0.5', mcalc)
    ok &= _testExpr('pyth(a, b) = sqrt(a^2 + b^2); pyth(3, 4)', '5', mcalc)

    # Test .del
    mcalc = MCalc()
    ok &= _testExpr('f(x) = x; .del f(x); .vars', 'last = 0', mcalc)

    if ok:
        print('All tests passed')


def _testExpr(expr, expectedResult, mcalc=None, expectError=False):
    if mcalc is None:
        mcalc = MCalc()
    results = [x for x in mcalc.calc(expr) if not x[1].startswith('Warning:')]
    if len(results) != 1:
        print('Test failure on "%s": expected one result, but got %d' % (expr, len(results)))
        return False
    isResult, result = results[0]
    if expectError:
        if isResult:
            print('Test failure on "%s": expected an error, but got result %s' % (expr, result))
            return False
    elif not isResult:
        print('Test failure on "%s": expected a result, but got %s' % (expr, result))
        return False
    if result != expectedResult:
        print('Test failure on "%s": expected %s but got %s' % (expr, expectedResult, result))
        return False
    return True


_helpTexts = dict(
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

    .del <var|fun>      Deletes a variable or function.
    
    .clear              Deletes all variables.
    
    .exit | .quit       Exits mcalc. The shortcut Ctrl-D also works.
    
    .reset              Deletes all variables and restores the default
                        settings.
    
    .vars               Shows all variables and their values, and all
                        user-defined functions and their definitions.
""",
        constants="""
The following constants are available. You can use them in mathematical
expressions like you would a variable.

    _c                  Speed of light in m/s, 299792458.

    e                   Euler's constant, ~2.718.
    
    i                   Imaginary unit, with i² = -1.
    
    pi, π               π is the ratio of a circle's circumference to its
                        diameter. ~3.142
    
    tau, τ              2π. ~6.283

Note that it is possible to create variables with the same name as these
constants. If you do so, the constants will not be accessible any more
until you delete those variables using the .del command.
""",
        functions="""
The following functions are builtin:

    Basic:              abs(x), log(x), log(x, b), ln(x), log2(x),
                        sqrt(x), cbrt(x)

    Trigonometric and   sin(a), cos(a), tan(a), asin(r), acos(r), atan(r),
    hyperbolic:         atan2(y, x), sinc(a), sinh(x), cosh(x), tanh(x),
                        asinh(x), acosh(x), atanh(x)
    
    Rounding:           ceil(x), floor(x), round(x), frac(x), sign(x)
    
    Complex numbers:    re(c), im(c), conj(c)
    
    Other:              deg(a), rad(a), rand(), gamma(x), binom(n, k)
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
To assign a value to variable, use the equals sign. You can then use the
variable in mathematical expressions. For example:

> x = 42
> 3x^2
                    5292
                    
You may also define functions like so:

> pythagoras(a, b) = sqrt(a^2 + b^2)
> pythagoras(3, 4)
                       5

Variable and function names may contain the letters from a-z in lower- or
uppercase, digits (except as first character) and the characters _ and '.
To see the list of currently defined variables, use the .vars command.
To delete variables, use the .del command.

The variable "last" will automatically be assigned the result of the last
calculation.       
""")


if mp.libmp.BACKEND != 'gmpy':
    _helpTexts['main'] += """
Note: no gmpy module is available. mcalc still works, but it will run much
faster with gmpy or gmpy2 available.
"""


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='%(prog)s is a an arbitrary precision command line calculator. '
                                         'Type "help" at the program\'s prompt for help on how to use it.',
                                         prog=_progName)
        parser.add_argument('-n', action='store_true', help="Don't run the RC file on startup. "
                            "The RC file is " + _rcFile)
        parser.add_argument('-t', action='store_true', help='run tests (intended for development)')
        parser.add_argument('-v', action='version', version='%(prog)s 0.99')
        args = parser.parse_args()

        if args.t:
            runTests()
        else:
            calculator = MCalc()
            calculator.run(args.n)
    except KeyboardInterrupt:
        sys.exit(-1)
