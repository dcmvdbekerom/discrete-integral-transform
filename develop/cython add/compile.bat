cls
::set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.25.28610\bin\Hostx64\x64
cython cython_add.pyx -a
python setup.py build_ext -i --force clean
pause