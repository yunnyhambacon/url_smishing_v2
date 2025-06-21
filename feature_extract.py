# feature_extract.py
import re
import warnings
from urllib.parse import urlparse
import tldextract

warnings.filterwarnings("ignore")   # tldextract 캐시 경고 억제용

# 안전한 urlparse 래퍼
def safe_urlparse(raw: str):
    """urlparse가 ValueError를 내면 대괄호 제거 후 재시도, 최종 실패시 dummy URL 반환"""
    url = str(raw).strip()
    if not re.match(r'^\w+://', url):
        url = 'http://' + url
    try:
        return urlparse(url)
    except ValueError:
        url_fixed = re.sub(r'[\[\]]', '', url)
        try:
            return urlparse(url_fixed)
        except ValueError:
            return urlparse('http://invalid.local/')

# 실제 Flask → 모델 전처리 함수
def parse_url_features(url: str) -> dict:
    parsed = safe_urlparse(url)

    # 1) 스킴 정보
    scheme = parsed.scheme.lower()
    scheme_https = int(scheme == 'https')
    scheme_http  = int(scheme == 'http')

    # 2) www 유무
    netloc = parsed.netloc.lower()
    has_www = int(netloc.startswith('www.'))

    # 3) 안전한 ccTLD 체크
    suffix = tldextract.extract(netloc or parsed.path).suffix.lower()
    safe_suffixes = {'co.kr','go.kr','or.kr','ac.kr','kr'}
    safe_ccTLD = int(suffix in safe_suffixes)

    # 4) URL 클린: 스킴·www 제거
    clean_part = (netloc + parsed.path).lstrip('www.') or str(url)

    # 5) 도메인 (모델 학습 시에도 쓰셨다면)
    domain = ".".join(p for p in [tldextract.extract(netloc).domain,
                                   tldextract.extract(netloc).suffix] if p)

    # 6) 기본 숫자 특성 (Lightning step)
    return {
        'url_length':   len(url),
        'num_dots':     url.count('.'),
        'num_slashes':  url.count('/'),
        'scheme_https': scheme_https,
        'scheme_http':  scheme_http,
        'has_www':      has_www,
        'safe_ccTLD':   safe_ccTLD,
        'URL_clean':    clean_part,
        'domain':       domain
    }
