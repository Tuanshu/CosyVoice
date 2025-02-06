    # "tensorrt-cu12==10.0.1",
    # "tensorrt-cu12-bindings==10.0.1",
    # "tensorrt-cu12-libs==10.0.1",

    https://github.com/astral-sh/uv/issues/6250

    這三個也會需要獨立安裝



    https://github.com/astral-sh/uv/issues/7347



    uv pip install WeTextProcessing==1.0.3 --no-build-isolation


     error: command '/usr/bin/x86_64-linux-gnu-g++' failed with exit code 1 遇到跟grpcio相同問題
     應該是跟python dev有關

    https://github.com/astral-sh/uv/issues/9347


    dpkg -l | grep python3

有ii  python3.12-dev                  3.12.9-1+jammy1                             amd64        Header files and a static library for Python (v3.12)

但仍然error: command '/usr/bin/x86_64-linux-gnu-g++' failed with exit code 1 

https://stackoverflow.com/questions/26053982/setup-script-exited-with-error-command-x86-64-linux-gnu-gcc-failed-with-exit


apt update && apt install -y libfst-dev


https://forums.developer.nvidia.com/t/cant-seem-to-get-anything-tts-or-asr-related-working-on-orin-nano/256695




改用uv pip install WeTextProcessing --no-build-isolation      也就是不強制1.0.3, which 需要pynini 2.1.5
這樣就算沒有apt update && apt install -y libfst-dev 也OK

 + importlib-resources==6.5.2
 + pynini==2.1.6
 + wetextprocessing==1.0.4.1

git submodule init

git submodule update --recursive
