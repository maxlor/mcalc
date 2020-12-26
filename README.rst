mcalc
=====

**An easy to use arbitrary precision command line calculator.**

Overview
--------

mcalc lets you type in mathematical expressions and then calculates their
value for you. For example::

    > 7*11*13
                        1001
    > 2 + 3 * 4 + 5
                          19
    > pi
                           3.141592654
    > asin acos atan tan cos sin 9
                           9

You can increase the display precision to whatever you need::

    > digits: 50
    > sqrt 2
                           1.4142135623730950488016887242096980785696718753769
    > digits:1001
    > 450!
        1733368733112632659344713146104579399677811265209051015569207509555333001683
    43675060467508829043871061458112845184240978586185838063016502083472961813516675
    70171918700422280962237272230663528084038062312369342674135036610101508838220494
    97092973901163679376616502373085389640390159083614414959443268420451378471640230
    31826040946839933150613025639183853033415106067614624202058200069363520959674171
    83191538725617509521380556781309195429800229273803342553558164591996298912368598
    54777117915846135134006890564712765816483637712630377492336007807230746200855435
    50683614481266062811457609604991878134283979248405925045378494874250604884810365
    71447957046788635742936714615176219148469743102979949740714485104716169664052397
    39260284840869400740899890112749290517151447343138663339249204066152269230304381
    39605419660932242438092251372688517179043032140582384479361116785682369730362384
    04626507890688000000000000000000000000000000000000000000000000000000000000000000
    000000000000000000000000000000000000000000000

Actually, you can independently set the calculation precision (using
``precision:``), even to a very high number of digits, like a million. You're
limited only by the amount of memory in your computer, and your patience,
since there might be slight slowdown at 10 million digits and higher.

mcalc lets you define variables and functions to use in your calculations::

    > x=4
    > pythagoras(a, b) = sqrt(a^2 + b^2)
    > pythagoras(3, x)
                           5
    > .vars
        last = 5
           x = 4

        pythagoras(a, b) = sqrt(a^2 + b^2)

(``last`` is an automatic variable, it contains the result of the last
computation.)

For **more information** on usage, type ``help`` at the mcalc prompt.

Installation
------------

For Windows users that don't have Python installed, I recommend using the
mcalc installer exe. It bundles mcalc, Python and the required modules.

If you already have Python, use the following steps:

**Step 1:** Download and save ``mcalc.py`` to wherever is convenient for you.
On unix, ``~/bin`` or ``/usr/local/bin`` might be suitable. Also on unix, you
might need to set the executable bit::

    $ chmod +x mcalc.py

**Step 2:** You'll also need the Python modules ``appdirs``, ``mpmath`` and
``rply``. They're easily installed using ``pip3``::

    $ pip3 install appdirs mpmath rply

**Step 2 Alternative:** If you're using an Ubuntu-like system, you could also
install the packages through ``apt``::

    $ sudo apt install python3-appdirs python3-mpmath python3-rply

**Optional:** mcalc supports an RC file, i.e. a file that is automatically
executed when mcalc is launched. If you put your custom functions in
there, they'll be available every time you run mcalc. The location is:

    Linux:
        ``~/.config/mcalc/mcalc.rc``
    MacOS:
        ``~/Library/Application Support/mcalc/mcalc.rc``
    Windows:
        ``C:\Users\<username>\AppData\Local\mcalc\mcalc.rc``

This is just a plain text file. You can put anything in there that you can
enter at the mcalc prompt.

Version History
---------------
    1.1.0 (2020-12-26)
	    Several fixes to expression parsing.

    1.0.3 (2020-10-28)
        Add a binary installer for Windows.

        Fix crash when calling an undefined function.

    1.0.2 (2020-08-20)
        Fix some bugs (wrong parsing of “-1-1”, division by 0 not handled).

        Add “π” and “τ” as alternatives to “pi” and “tau”.

    1.0.1 (2020-06-03)
        Make readline optional because it may be unavailable on Windows.

    1.0 (2020-06-02)
        Initial release.

License
-------
mcalc is Copyright 2020 Benjamin Lutz.

mcalc is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License version 3 as published by the
Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details.

You can find a copy of the GNU General Public License in the file LICENSE or
at https://www.gnu.org/licenses/gpl-3.0.html.
