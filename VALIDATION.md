# 검증 기록

검증일: 2026-10-15

## 환경

- Windows / Anaconda 기반 Python 3.11.17 / 프로젝트 `.venv`
- RTX 3080 12GB / 시스템 RAM 32GB
- PyTorch 2.11.0+cu128, PyQt6 6.11.0, faster-whisper 1.2.1, Transformers 4.57.6
- 전체 패키지 버전: `requirements-lock.txt`

## 통과

- `pytest`: 10개 테스트. SRT/VTT 시간·한국어 보존, 부분 저장, 잘못된 시간 검증, 재시도 시 완료 구간 유지, STT 캐시 재사용, 취소 후 중간 결과 보존, 입력 분할 시 원문 보존, FFmpeg 정규화, UI 수정·파일 저장을 확인했다.
- `pip check`: 패키지 의존성 충돌 없음.
- `run.bat --smoke-test`: 가상환경 활성화 및 실제 Qt 이벤트 루프 실행 후 정상 종료.
- `check_environment.py`: CUDA 인식 및 bitsandbytes NF4 GPU 연산 성공.
- 공식 배포 Deno v2.9.7 다운로드의 SHA-256 검증 및 실행 성공.
- 공개 19초 샘플 영상 `jNQXAC9IVRw`: 실제 YouTube 메타데이터 조회, 영상 다운로드, 영상/음성 트랙 및 첫 프레임 디코딩 확인.
- 같은 영상: CPU/int8 Whisper tiny로 실제 전사 후 원본 SRT 저장.
- 같은 영상: 기존 영어 수동 자막 다운로드 및 SRT 변환, 6개 구간 파싱 성공.
- Qt worker/controller 전체 흐름: 영상 다운로드 → 오디오 정규화 → 실제 STT → 동일 언어 번역 생략 → 원본/결과 자막 저장 성공.
- 분리한 테스트 영상·음성을 yt-dlp의 FFmpeg 병합기로 합치고 최종 MP4의 영상·음성 트랙 및 프레임 디코딩 확인.
- UI 이미지를 렌더링하여 설정·실행·자막 조회·내보내기 영역을 확인했다.

## 아직 검증하지 못한 항목

**TranslateGemma 12B 실제 모델 로딩 및 언어 간 번역 품질**: 이 PC에 Hugging Face 로그인 정보가 없으며 모델 설정 파일 요청이 HTTP 401로 거부되었다. 사용자가 모델 페이지에서 이용 조건에 동의하고 `login_huggingface.bat`으로 로그인한 뒤 확인해야 한다. 작은 NF4 텐서 연산 성공은 12B 전체 모델의 VRAM 적합성이나 번역 성공을 보장하지 않는다.

긴 영상 처리 성능과 번역 품질 검토도 실제 모델 접근 후 수행한다. STT 통합 검증에는 빠른 검증을 위해 tiny를 사용했으며 앱 기본값은 small이다.

## 검증 중 반영한 수정

- Windows에서 PyTorch보다 PyAV를 먼저 초기화하여 PyAV 필터 DLL import 정지를 방지했다.
- 모델 다운로드에 일반 로컬 폴더를 사용하여 관리자 권한이 없는 Windows에서의 심볼릭 링크 권한 및 초기 병렬 다운로드 문제를 피했다.

## 후속 UI 및 CUDA 수정 검증

- LLM ID Path, 세션 전용 Access 토큰, 번역 준비사항 검사와 독립적인 LLM 모델 다운로드 버튼을 추가했다.
- 모델 다운로드는 모의 Hub 응답으로 토큰 전달, 고정 리비전, 완료 파일 재사용, 파일 간 취소, 영상·GPU와 독립적인 실행을 확인했다. 인증 정보가 없어 실제 12B 다운로드는 아직 검증하지 못했다.
- RTX 3080에서 Whisper tiny를 **CUDA/float16**으로 실행하여 같은 19초 오디오를 실제 전사했고 영어 2개 구간을 얻었다. GPU 전사 성공 및 CUDA 미지원/초기 실패 시 CPU 전환을 회귀 테스트로 확인했다.
