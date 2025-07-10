# 동기화 알고리즘의 구현과 검증
# 알고리즘 설명:
# 1. 해시테이블(defaultdict)을 사용하여 content_hash에서 chunks로의 매핑을 구축, 시간복잡도 O(n)
# 2. 집합 연산을 사용하여 동일한 위치의 chunks를 찾음, 시간복잡도 O(n)
# 3. 두 포인터 방법을 사용하여 나머지 chunks를 매칭, 시간복잡도 O(n)
# 전체 시간복잡도: O(n), 여기서 n은 chunks의 총 개수
# 공간복잡도: O(n), 주로 해시테이블 저장에 사용

from collections import defaultdict
from typing import TypedDict, List, Dict, Set
from dataclasses import dataclass

@dataclass
class Chunk:
    index: int
    content_hash: str
    chunk_content: str
    uuid: str = None

class SyncResult(TypedDict):
    to_create: List[Dict]
    to_update: List[Dict]
    to_delete: List[str]

# 백엔드의 기존 chunks 데이터 시뮬레이션
old_chunks = [
    {'uuid': 'uuid_1', 'index': 0, 'content_hash': 'hash_A', 'chunk_content': '첫 번째 단락입니다.'},
    {'uuid': 'uuid_2', 'index': 1, 'content_hash': 'hash_B', 'chunk_content': '두 번째 단락입니다.'},
    {'uuid': 'uuid_3', 'index': 2, 'content_hash': 'hash_C', 'chunk_content': '세 번째 단락입니다.'},
    {'uuid': 'uuid_4', 'index': 3, 'content_hash': 'hash_D', 'chunk_content': '네 번째 단락입니다.'},
    {'uuid': 'uuid_5', 'index': 4, 'content_hash': 'hash_E', 'chunk_content': '다섯 번째 단락입니다.'},
]

# GitHub Actions에서 생성된 새로운 chunks 데이터 시뮬레이션
new_chunks = [
    {'index': 0, 'content_hash': 'hash_A', 'chunk_content': '첫 번째 단락입니다.'},
    {'index': 1, 'content_hash': 'hash_C', 'chunk_content': '세 번째 단락입니다.'},
    {'index': 2, 'content_hash': 'hash_D', 'chunk_content': '네 번째 단락입니다.'},
    {'index': 3, 'content_hash': 'hash_D', 'chunk_content': '네 번째 단락입니다.'},
    {'index': 4, 'content_hash': 'hash_D', 'chunk_content': '네 번째 단락입니다.'},
    {'index': 5, 'content_hash': 'hash_D', 'chunk_content': '네 번째 단락입니다.'},
    {'index': 6, 'content_hash': 'hash_D', 'chunk_content': '네 번째 단락입니다.'},
]

def synchronize_chunks(old_chunks: List[Dict], new_chunks: List[Dict]) -> SyncResult:
    """
    content_hash + index 기반의 두 포인터 매칭 알고리즘으로, 추가, 업데이트, 삭제가 필요한 chunks를 찾습니다.
    주요 개선사항:
    1. 동일한 content_hash의 기존, 새로운 chunks를 각각 index로 정렬한 후 개별 매칭하여, 
       기존의 "두 개씩 동일한 위치"에 의한 중복 content_hash 시 혼란을 방지합니다.
    2. 기존의 거리 임계값(distance <= threshold) 판단을 유지하되, 로직을 더 직관적으로 하여 
       누락 매칭이나 오판을 줄입니다.
    """

    # ========== 1. 입력 검증 ==========
    if not isinstance(old_chunks, list) or not isinstance(new_chunks, list):
        raise TypeError("입력 매개변수는 리스트 타입이어야 합니다")

    required_fields = {'index', 'content_hash', 'chunk_content'}
    for chunk in old_chunks:
        if not required_fields.union({'uuid'}).issubset(chunk.keys()):
            raise ValueError("기존 chunks에 필수 필드가 누락되었습니다")
    for chunk in new_chunks:
        if not required_fields.issubset(chunk.keys()):
            raise ValueError("새로운 chunks에 필수 필드가 누락되었습니다")

    # ========== 2. content_hash => chunks 매핑 테이블 구축, content_hash 간 오류 매칭 감소 ==========
    old_chunks_by_hash = defaultdict(list)
    for oc in old_chunks:
        old_chunks_by_hash[oc['content_hash']].append(oc)

    new_chunks_by_hash = defaultdict(list)
    for nc in new_chunks:
        new_chunks_by_hash[nc['content_hash']].append(nc)

    # ========== 3. 모든 content_hash를 순회하며 개별 매칭 ==========

    to_create = []
    to_update = []
    to_delete = []

    # "합집합"으로 모든 content_hash 획득
    all_hashes = set(old_chunks_by_hash.keys()) | set(new_chunks_by_hash.keys())

    # 허용되는 업데이트 거리 임계값, 필요에 따라 조정 가능
    threshold = 10

    for content_hash in all_hashes:
        old_list = sorted(old_chunks_by_hash[content_hash], key=lambda x: x['index'])
        new_list = sorted(new_chunks_by_hash[content_hash], key=lambda x: x['index'])

        i, j = 0, 0
        len_old, len_new = len(old_list), len(new_list)

        while i < len_old and j < len_new:
            old_entry = old_list[i]
            new_entry = new_list[j]
            distance = abs(old_entry['index'] - new_entry['index'])

            # 如果索引相近，则判定为同一块内容，执行更新操作
            if distance <= threshold:
                to_update.append({
                    'uuid': old_entry['uuid'],
                    'index': new_entry['index'],
                    'content_hash': content_hash,
                    'chunk_content': new_entry['chunk_content']
                })
                i += 1
                j += 1

            # 如果旧 chunk.index 更小，说明它在新列表里没有合适的配对，需要删除
            elif old_entry['index'] < new_entry['index']:
                to_delete.append(old_entry['uuid'])
                i += 1

            # 否则，新 chunk.index 更小，说明这是新增加的块
            else:
                to_create.append({
                    'index': new_entry['index'],
                    'content_hash': content_hash,
                    'chunk_content': new_entry['chunk_content']
                })
                j += 1

        # 把剩余的旧 chunks 视为需要删除
        while i < len_old:
            to_delete.append(old_list[i]['uuid'])
            i += 1

        # 把剩余的新 chunks 视为需要新增
        while j < len_new:
            to_create.append({
                'index': new_list[j]['index'],
                'content_hash': content_hash,
                'chunk_content': new_list[j]['chunk_content']
            })
            j += 1

    return {
        'to_create': to_create,
        'to_update': to_update,
        'to_delete': to_delete
    }

if __name__ == '__main__':
    result = synchronize_chunks(old_chunks, new_chunks)

    print("생성해야 하는 chunks:")
    if result['to_create']:
        for chunk in result['to_create']:
            print(chunk)
    else:
        print("null")

    print("\n업데이트해야 하는 chunks:")
    if result['to_update']:
        for chunk in result['to_update']:
            print(chunk)
    else:
        print("null")

    print("\n삭제해야 하는 chunks:")
    if result['to_delete']:
        for uuid in result['to_delete']:
            print(uuid)
    else:
        print("null")