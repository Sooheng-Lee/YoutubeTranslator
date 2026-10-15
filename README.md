# YouTube Translator

Python / PyQt6 데스크톱 앱. faster-whisper로 원어를 전사한 뒤 **TranslateGemma 12B**를 Transformers로 직접 실행합니다. Ollama는 사용하지 않습니다.

## 처음 사용하는 경우

### 1. 설치 및 실행

1. Windows PC에 Anaconda 또는 Miniconda를 준비합니다. 설치 스크립트는 `%USERPROFILE%\anaconda3` 또는 `%USERPROFILE%\miniconda3`에서 찾습니다. 다른 위치에 설치했다면 `setup.bat`의 `CONDA_EXE_PATH`를 해당 위치로 수정합니다.
2. 프로젝트 폴더의 `setup.bat`을 더블클릭하고 설치가 끝날 때까지 기다립니다. 인터넷 연결이 필요합니다. Anaconda에 Python 3.11 기반 환경을 만들고 앱 패키지는 프로젝트의 `.venv`에 설치합니다.
3. `Setup complete`가 표시되면 `run.bat`을 더블클릭합니다. 이미 환경을 구성했다면 설치를 반복하지 않고 `run.bat`만 실행하면 됩니다.

PowerShell에서도 실행할 수 있습니다.

```powershell
cd D:\YouTubeTranslator
.\setup.bat  # 최초 설치 때 실행
.\run.bat    # 이후 프로그램 실행
```

`run.bat`은 `.venv`를 활성화한 뒤 `main.py`를 실행합니다. 시스템 Python이나 Anaconda base에 앱 모듈을 따로 설치할 필요는 없습니다.

### 2. LLM 모델 다운로드

1. [TranslateGemma 12B 모델 페이지](https://huggingface.co/google/translategemma-12b-it)에 Hugging Face 계정으로 로그인하고 모델 이용 조건에 동의합니다.
2. 앱의 **LLM ID Path**에 `google/translategemma-12b-it`를 입력합니다. 기본값으로 설정되어 있습니다.
3. 모델을 읽을 수 있는 Hugging Face 토큰을 **Access 토큰**에 입력합니다. `login_huggingface.bat`으로 이미 로그인했다면 입력을 생략할 수 있습니다.
4. 입력란 옆의 **LLM 모델 다운로드** 버튼을 누릅니다.
5. 상태 영역에서 파일별 진행 상황을 확인합니다. 완료되면 모델 저장 경로가 표시됩니다.

모델 다운로드에는 영상 URL과 GPU가 필요하지 않습니다. 원본 모델 파일 기준으로 수십 GB를 받으므로 저장 공간과 다운로드 시간을 확보하세요. `4bit`는 실행 시 가중치를 양자화하는 설정이며 다운로드 파일 크기를 줄이지 않습니다.

진행률은 바이트가 아닌 **완료된 파일 수 기준**입니다. 큰 가중치 파일을 받는 동안 같은 진행률이 오래 표시될 수 있습니다. **취소**를 누르면 현재 파일 다운로드가 끝난 뒤 중단하며, 다시 실행하면 이미 받은 파일을 재사용합니다.

토큰은 비밀번호처럼 가려지고 현재 앱 세션에서만 사용합니다. 설정·작업 파일에 저장하지 않으며 앱을 종료하면 다시 입력해야 합니다. 모델 다운로드가 모두 끝났다면 같은 PC에서 해당 모델을 사용할 때 토큰을 생략할 수 있습니다. 현재 다운로드 및 번역 엔진은 TranslateGemma 12B를 지원합니다.

### 3. 영상 번역

1. **YouTube URL**에 단일 영상 주소를 붙여넣습니다.
2. 원본 언어를 **자동 감지** 또는 영상의 실제 언어로 지정하고 목표 언어를 선택합니다. 예: 영어 영상 → 한국어.
3. **STT 모델**을 선택합니다. 기본값은 `small`입니다. `tiny`/`base`는 빠른 확인용으로, 큰 모델은 더 많은 자원을 사용하므로 필요에 맞게 선택합니다.
4. **번역 실행 장치 / 정밀도**를 지정합니다. 검증한 RTX 3080 12GB 환경에서는 **cuda / 4bit**를 사용합니다.
5. **저장 폴더**를 선택합니다. 영상도 필요하면 **번역 시 영상도 함께 저장**을 체크합니다.
6. **번역 시작**을 누릅니다.
7. 준비 확인 → 다운로드 → 오디오 전처리 → STT → 번역 순서로 진행 상황을 확인합니다. STT 모델도 처음 사용할 때 다운로드합니다.
8. 완료 후 표에서 원문과 번역문을 비교합니다. 번역문은 더블클릭하여 수정할 수 있습니다.

번역 시작 전 필수 모듈, FFmpeg, JavaScript 런타임, 모델 파일/접근 권한, 저장 공간, 실행 장치와 메모리를 검사합니다. 부족한 항목은 오류창으로 안내하며 영상 다운로드와 STT를 시작하지 않습니다. 모델 다운로드가 끝나도 GPU 실행 조건이 부족하면 번역을 시작할 수 없습니다.

STT 장치는 자동 선택합니다. CUDA와 실행 라이브러리가 준비되어 있으면 GPU를 사용하고 그렇지 않으면 CPU/int8로 전환합니다. 상태 영역에 실제 장치가 표시됩니다. 번역 장치 설정과는 독립적입니다.

### 4. 자막 저장 및 수정

1. 결과 표의 **번역문 (수정 가능)** 칸을 더블클릭하여 필요한 문장을 수정합니다. 수정 내용은 작업 기록에 저장됩니다.
2. 왼쪽 아래에서 `srt`, `vtt`, `txt` 중 출력 형식을 선택합니다.
3. **원본 자막 저장**, **번역 자막 저장**, **두 자막 함께 저장** 중 필요한 버튼을 누릅니다.
4. **저장 폴더 열기**로 결과 파일을 확인합니다. 기존 자막 파일과 이름이 같으면 덮어쓰기 여부를 확인합니다.

여기서 원본 자막은 **Whisper가 음성에서 생성한 전사 결과**입니다. STT가 끝났다면 번역이 진행 중이거나 실패한 상태에서도 원본 자막을 저장할 수 있습니다. 번역 자막에 미완료 구간이 있으면 해당 개수를 표시하고 부분 저장 여부를 확인합니다. TXT는 구간별 원문과 번역문을 함께 저장합니다.

예를 들어 영어를 한국어로 번역하면 다음처럼 저장됩니다.

```text
<영상ID>.en.stt.srt           # Whisper 원본 자막
<영상ID>.ko.translated.srt    # 수정 사항을 반영한 한국어 번역 자막
```

## 영상이나 기존 YouTube 자막만 다운로드

### 영상만 저장

1. 영상 URL과 저장 폴더를 입력합니다.
2. **영상 정보 / 기존 자막 조회**를 누르면 제공 화질을 확인할 수 있습니다.
3. 최고 화질 또는 최대 해상도를 선택하고 **영상만 다운로드**를 누릅니다. 선택값 이하에서 제공되는 화질을 사용합니다.
4. 다운로드·병합이 끝난 뒤 저장 폴더에서 영상 파일을 확인합니다.

음성이 포함된 MP4를 우선하며 코덱 조합에 따라 MKV로 병합할 수 있습니다. 영상 다운로드에는 LLM 토큰이나 모델 로딩이 필요하지 않습니다. 기존 최종 영상은 덮어쓰지 않고 재사용합니다.

### 기존 YouTube 자막 저장

1. 영상 URL을 입력하고 **영상 정보 / 기존 자막 조회**를 누릅니다.
2. 자막 목록에서 언어와 **수동 / 자동 생성** 구분을 확인하고 선택합니다.
3. 출력 형식을 `srt` 또는 `vtt`로 선택합니다.
4. **선택한 YouTube 자막 저장**을 누르고 파일 위치를 지정합니다.

자막이 없는 영상은 사용할 수 없다는 안내가 표시됩니다. 이 경우 번역 기능으로 Whisper 원본 자막을 생성할 수 있습니다. 기존 YouTube 자막은 파일로 저장하는 기능이며 번역 입력은 계속 Whisper 전사 결과를 사용합니다.

## 취소 및 실패 후 재시도

처리 중 **취소**를 누르면 안전한 중단 지점에서 멈춥니다. 모델 파일 다운로드는 현재 파일 완료까지, 모델 로딩과 STT는 현재 내부 연산이 끝날 때까지 기다릴 수 있습니다. 취소 요청 뒤 종료를 기다려 주세요.

같은 영상과 설정으로 **미완료 구간 재시도**를 누르면 완료된 STT와 번역을 재사용하고 미완료 구간을 다시 처리합니다. 앱을 재시작한 경우 URL·언어·모델·정밀도를 이전과 같게 지정한 뒤 다시 시작하세요. 저장된 결과를 찾아 이어서 처리합니다. 앱 실행 직후 이전 결과를 자동으로 표에 표시하지는 않습니다.

목표 언어·LLM·정밀도를 바꾸면 번역 작업이 달라집니다. STT 모델이나 원본 언어를 바꾸면 STT부터 새로 처리할 수 있습니다. 완료된 번역과 수동 수정은 동일한 설정의 재시도에서 유지됩니다.

## 자주 발생하는 문제

| 증상 | 확인 및 조치 |
| --- | --- |
| 가상환경이 없다는 메시지 | `setup.bat`으로 환경을 구성한 뒤 `run.bat` 실행 |
| Anaconda/Miniconda를 찾지 못함 | 설치 위치를 확인하고 `setup.bat`의 `CONDA_EXE_PATH` 수정 |
| Access 토큰 필요 / HTTP 401·403 | 모델 이용 조건 동의, 토큰 읽기 권한 확인 후 다시 입력하거나 `login_huggingface.bat`으로 로그인 |
| LLM ID Path 오류 | 기본 모델 ID 또는 완전한 TranslateGemma 12B 로컬 폴더 지정. 다른 LLM의 번역은 현재 지원하지 않음 |
| 모델 다운로드 진행률이 오래 그대로임 | 큰 파일 다운로드 중일 수 있음. 현재 파일명과 네트워크 상태 확인. 진행률은 완료 파일 수 기준 |
| GPU 메모리 부족 | 다른 GPU 사용 프로그램을 종료하고 `cuda / 4bit` 선택. 긴 입력과 실행 환경에 따라 추가 메모리가 필요할 수 있음 |
| CUDA 또는 BF16 미지원 | `check_environment.py`로 GPU와 CUDA PyTorch 확인. LLM의 CUDA 실행은 BF16 지원이 필요하며 CPU/float32는 충분한 RAM이 필요함 |
| 번역 중 출력 길이 초과·NaN/Inf·PAD 반복 | 앱을 완전히 종료하고 최신 코드로 재실행한 뒤 `cuda / 4bit`로 재시도. LLM은 BF16 계산을 사용해야 함 |
| 연속 3개 구간 실패로 중단 | 오류창·상태 영역의 원인을 확인하여 해결한 뒤 미완료 구간 재시도. 표의 실패 셀에 마우스를 올리면 구간 오류도 확인 가능 |
| Whisper가 CPU로 실행됨 | GPU와 cuBLAS/cuDNN을 사용할 수 없거나 CUDA 초기 실행에 실패했는지 상태 메시지 확인. CPU로 계속 전사 가능 |
| YouTube 추출 실패 | URL·네트워크·영상 접근 가능 여부 확인. 필요하면 아래 명령으로 yt-dlp와 EJS를 함께 업데이트 |
| 기존 YouTube 자막이 없음 | 제공 자막 다운로드 대신 Whisper 전사로 원본 자막 생성 |

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "yt-dlp[default]"
.\.venv\Scripts\python.exe check_environment.py
```

번역 오류가 계속되면 상태 영역의 실패 원인과 `%LOCALAPPDATA%\YouTubeTranslator\jobs\`의 해당 작업 JSON에 있는 `error` 항목을 확인하세요. 문제를 공유할 때 Access 토큰을 포함하지 마세요.

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
