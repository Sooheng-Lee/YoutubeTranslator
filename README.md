# YouTube Translator

Python / PyQt6 데스크톱 앱. faster-whisper로 원어를 전사한 뒤 **TranslateGemma 12B**를 Transformers로 직접 실행합니다. Ollama는 사용하지 않습니다.

## 실행

1. `run.bat`을 더블클릭합니다. 프로젝트의 `.venv`를 활성화하고 `main.py`를 실행합니다.
2. YouTube URL을 입력하고 원본/목표 언어 및 저장 폴더를 선택합니다.
3. **LLM ID Path**에 모델 ID 또는 로컬 경로를 지정합니다. 옆의 **Access 토큰**에 Hugging Face 읽기 토큰을 입력할 수 있습니다. 입력값은 가려지며 현재 세션에서만 사용하고 설정·작업 파일에 저장하지 않습니다. **LLM 모델 다운로드** 버튼으로 모델을 미리 받을 수 있습니다. 영상 URL이나 CUDA는 필요하지 않으며 완료 시 모델 저장 경로를 표시합니다. 진행률은 완료 파일 수 기준이고, 취소는 현재 파일 다운로드가 끝난 뒤 적용됩니다. 현재 지원 엔진과 다운로드 모델은 TranslateGemma 12B입니다.
4. **번역 시작**을 누르면 필수 모듈, FFmpeg, JavaScript 런타임, 모델 파일 또는 다운로드 접근 권한, 저장 공간, CUDA/정밀도와 메모리를 먼저 확인합니다. 부족한 항목은 오류창으로 안내하고 영상 다운로드·STT를 시작하지 않습니다. 원격 검사는 네트워크에 따라 시간이 걸릴 수 있으며 UI는 계속 응답합니다. STT 모델과 번역 모델은 첫 사용 때 다운로드됩니다.
5. 번역문은 표에서 더블클릭하여 수정합니다. 원본 자막·번역 자막을 각각 또는 함께 저장할 수 있습니다.

영상 다운로드만 필요하면 **영상만 다운로드**를 사용합니다. 모델을 로딩하지 않습니다. **영상 정보 / 기존 자막 조회**로 제공 화질 및 기존 자막을 조회하고 언어와 수동/자동 생성 자막을 선택할 수 있습니다. 기존 자막 저장은 STT 기반 번역과 독립적입니다.

## 환경 구성

- Anaconda Python 환경: `youtube-translator-bootstrap` (Python 3.11)
- 프로젝트 가상환경: `.venv/`
- Conda 환경에는 Python과 그 기본 의존성만 설치합니다. 앱 모듈은 `.venv`에 설치합니다.
- 새로 구성하려면 `setup.bat`을 실행합니다. `%USERPROFILE%\anaconda3` 또는 `miniconda3`를 찾습니다.
- Anaconda 기본 채널 약관을 자동 수락하지 않으며 conda-forge를 사용합니다.
- CUDA PyTorch 설치: `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- 나머지 패키지: `python -m pip install -r requirements.txt`
- 현재 설치 버전 스냅샷은 `requirements-lock.txt`에 기록합니다. 재구성 시 CUDA torch/torchvision은 위 인덱스에서 먼저 설치하고 `pip install -r requirements-lock.txt`를 실행합니다.
- FFmpeg는 `imageio-ffmpeg`에 포함된 실행 파일을 사용합니다. 시스템 PATH에 별도로 설치할 필요가 없습니다.
- YouTube 추출에 필요한 Deno는 `setup_runtime.py`가 공식 GitHub 배포 파일의 SHA-256을 검증하여 프로젝트 `.tools/`에 설치합니다. `yt-dlp-ejs`는 `yt-dlp[default]`와 함께 설치됩니다. [yt-dlp EJS 안내](https://github.com/yt-dlp/yt-dlp/wiki/EJS)

Anaconda의 Python 환경은 `.venv`의 기반 Python이므로 삭제하거나 이동하지 마세요. 프로젝트 폴더를 이동했다면 `.venv`를 다시 만들어야 합니다.

## TranslateGemma 접근 및 하드웨어

모델: `google/translategemma-12b-it`.

1. [공식 모델 페이지](https://huggingface.co/google/translategemma-12b-it)에 로그인하고 모델 이용 조건에 동의합니다.
2. UI의 **Access 토큰**에 읽기 권한 토큰을 입력하거나 `login_huggingface.bat`으로 로그인합니다. UI 토큰은 디스크에 저장하지 않으며, 비어 있으면 기존 Hugging Face 로그인 정보를 사용합니다. 완전한 로컬 모델 또는 앱 다운로드 폴더의 모든 모델 파일이 준비되어 있으면 로그인 없이 사용할 수 있습니다. 토큰을 소스 파일이나 채팅에 작성하지 마세요.
3. 앱에서 기본값 **cuda / 4bit**로 실행합니다. 가중치 다운로드는 4비트 실행이어도 원본 모델 크기 기준으로 수십 GB가 필요하며 첫 로딩에 시간이 걸립니다.

이 PC의 RTX 3080 12GB를 기준으로 NF4 4비트 저장 및 **BF16 계산**을 기본 설정했습니다. TranslateGemma의 BF16 가중치를 FP16으로 계산하면 PAD 토큰 반복과 출력 길이 초과가 발생할 수 있어, LLM에는 FP16을 사용하지 않습니다. 기존 float16 설정은 bfloat16으로 전환합니다. [Gemma 3 FP16 문제 기록](https://github.com/huggingface/transformers/issues/36822)

다른 프로그램의 GPU 사용량과 입력 길이에 따라 메모리가 부족할 수 있습니다. bfloat16/float32 전체 GPU 로딩은 더 큰 VRAM이 필요합니다. CPU float32 실행은 약 56GB 이상 여유 RAM이 있어야 시작하며 이 PC(32GB RAM)에서는 CUDA/4bit를 사용하세요. STT의 float16 설정은 LLM과 독립적입니다.

STT는 CUDA GPU, CTranslate2 지원 연산 형식, cuBLAS/cuDNN 라이브러리를 자동 확인하여 가능하면 GPU/float16으로 실행합니다. 라이브러리나 GPU를 사용할 수 없거나 CUDA 초기 실행에 실패하면 CPU/int8로 전환하며 상태 영역에 실제 장치를 표시합니다. CUDA 전사 도중 이미 자막 구간이 생성된 후 발생한 오류는 중복 결과를 막기 위해 오류로 안내하고 재시도하게 합니다. Windows에서는 설치된 PyTorch에 포함된 cuBLAS/cuDNN DLL을 사용합니다. STT 모델 기본값은 `small`이며 UI에서 변경할 수 있습니다. STT 장치 자동 선택은 번역 장치 설정과 독립적이고 STT 자원을 해제한 후 번역 모델을 로딩합니다.

로컬 모델 경로는 사용자가 준비한 TranslateGemma 12B 폴더를 지정합니다. 앱은 12B 아키텍처를 검증하지만 같은 구조의 다른 파인튜닝 모델과의 동일성을 보증하지 않으므로 모델 출처를 확인하세요.

## 결과 및 복구

- 기본 결과 폴더: 사용자 `Downloads/YouTubeTranslator`.
- 원본 자막: `<영상ID>.<원어>.stt.srt/vtt`.
- 번역 자막: `<영상ID>.<목표언어>.translated.srt/vtt`.
- 기존 YouTube 자막: `<영상ID>.<언어>.youtube-manual/auto.srt/vtt`.
- TXT는 구간별 원문과 번역문을 함께 저장합니다.
- 영상은 영상 ID 기반 파일명으로 저장하며, 분리 스트림을 병합합니다. MP4를 우선하고 호환되지 않는 조합은 MKV로 저장할 수 있습니다. 기존 최종 영상은 덮어쓰지 않고 재사용합니다.
- 번역이 실패해도 완료된 STT 원본을 저장할 수 있습니다. 번역 미완료 구간을 제외한 부분 저장은 확인 후 수행합니다.
- 설정, STT 및 번역 중간 결과: `%LOCALAPPDATA%/YouTubeTranslator/`.
- 모델도 위 폴더의 `models/`에 저장합니다. 관리자 권한이나 Windows 개발자 모드 없이 다운로드할 수 있도록 일반 파일로 저장합니다.
- 같은 영상·STT 설정은 완료된 STT를 재사용합니다. 같은 목표 언어·모델·정밀도의 완료 번역과 수동 수정은 재시도 때 유지됩니다. 앱 시작 시 자동 복원하지는 않지만 같은 URL/설정으로 시작하면 저장된 결과를 찾습니다.
- 번역 입력은 공식 템플릿의 실제 토큰 수로 검사하고 긴 구간을 분할합니다. 세그먼트 ID와 시간은 유지합니다.
- 생성 중 NaN/Inf 수치 오류는 즉시 중단하며 일반 오류도 3개 구간에서 연속 발생하면 중단하고 원인을 표시합니다. 기존 STT·완료 번역은 보존되므로 앱 재시작 후 같은 영상/설정으로 미완료 구간을 재시도할 수 있습니다.
- 취소는 다운로드 하위 프로세스와 해당 FFmpeg를 종료하고 추론은 토큰 생성 경계에서 중단합니다. 모델 다운로드/로딩 또는 STT의 현재 내부 연산은 즉시 취소되지 않을 수 있습니다.
- 임시 오디오와 중간 결과는 재시도를 위해 보관합니다. 최종 영상·자막은 앱이 자동 삭제하지 않습니다.

## 구조 및 검증

`main.py` → `AppController` → `PipelineWorker` / `TranslationPipeline` → 서비스 클래스 순서로 실행합니다. 긴 작업은 QThread에서 수행하고 UI에는 signal로 결과를 전달합니다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe check_environment.py
.\.venv\Scripts\python.exe main.py --smoke-test
```

실제 YouTube 접근은 네트워크·영상 상태 및 yt-dlp의 추출 지원 상황에 따라 달라집니다. 추출이 실패하면 가상환경의 yt-dlp를 업데이트한 뒤 재시도하세요. DRM·접근 제한 우회, 영상 합성, 더빙, 실시간 번역은 제공하지 않습니다.
