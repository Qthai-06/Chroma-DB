# -*- coding: utf-8 -*-
import csv  
from pathlib import Path 
from typing import Dict, Iterable, List, Tuple 
# Cài đặt thư viện chromadb nếu chưa có: pip install chromadb
import chromadb

BASE_DIR = Path(__file__).resolve().parent 
CSV_FILE = BASE_DIR / "data.csv"  # đg dẫn tới file CSV
COLLECTION_NAME = "qa_collection" # tên collection

ID_CANDIDATES = ("id", "person_id", "uid") # các tên cột khả dĩ cho ID
QUESTION_CANDIDATES = ("question", "title", "name") # các tên cột khả dĩ cho câu hỏi
ANSWER_CANDIDATES = ("answer", "ability", "description", "skill") # các tên cột khả dĩ cho câu trả lời

# Cline lưu trữ vĩnh viễn (tạo thư mục chromadb_store)
client = chromadb.PersistentClient(path=str(BASE_DIR / "chromadb_store")) 
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Thông tin cột được sử dụng trong quá trình nạp dữ liệu
COLUMN_INFO: Dict[str, str] = {}

# Hai hàm tiện ích: Chuẩn hóa tên cột và chọn cột hợp lệ
def _normalize_header(name: str | None) -> str:
    return (name or "").strip().lower()
def _resolve_field( 
    available_headers: Dict[str, str],
    candidates: Iterable[str],
    label: str,
) -> str:
    for candidate in candidates:
        normalized = _normalize_header(candidate)
        if normalized in available_headers:
            return available_headers[normalized]
    raise ValueError(
        f"Không tìm thấy cột phù hợp cho '{label}'. CSV cần một trong các cột: {', '.join(candidates)}."
    )

# Đọc CSV và trả về các tài liệu, metadata, ids và thông tin cột
def load_csv_rows(
    file_path: Path,
) -> Tuple[List[str], List[Dict[str, str]], List[str], Dict[str, str]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {file_path}")

    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []
    ids: List[str] = []

    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        normalized_headers = {_normalize_header(name): name for name in fieldnames}

        id_field = _resolve_field(normalized_headers, ID_CANDIDATES, "id")
        question_field = _resolve_field(normalized_headers, QUESTION_CANDIDATES, "question")
        answer_field = _resolve_field(normalized_headers, ANSWER_CANDIDATES, "answer")

        for row_index, row in enumerate(reader):
            answer_text = (row.get(answer_field) or "").strip()
            question_text = (row.get(question_field) or "").strip()
            if not answer_text and not question_text:
                # Bỏ qua dòng không có dữ liệu hữu ích
                continue

            raw_id = (row.get(id_field) or "").strip()
            doc_id = raw_id or f"row_{len(documents)}"

            documents.append(answer_text or question_text)
            metadata = {k: v for k, v in row.items() if k != answer_field}
            metadatas.append(metadata)
            ids.append(doc_id)

    if not documents:
        raise ValueError("Không có dòng nào hợp lệ trong CSV sau khi xử lý.")

    column_info = {"id": id_field, "question": question_field, "answer": answer_field}
    return documents, metadatas, ids, column_info

# Nạp dữ liệu từ CSV vào collection và in tóm tắt
docs, metas, doc_ids, COLUMN_INFO = load_csv_rows(CSV_FILE)
collection.upsert(documents=docs, metadatas=metas, ids=doc_ids)
if hasattr(client, "persist"):
    client.persist()
print(
    f"Đã nạp {len(doc_ids)} dòng từ {CSV_FILE}. "
    f"Sử dụng cột '{COLUMN_INFO['question']}' làm câu hỏi và '{COLUMN_INFO['answer']}' làm câu trả lời."
)

# Chế độ hỏi-đáp tương tác
def interactive_search():
    print("Nhập câu hỏi (gõ 'exit' để thoát)")
    while True:
        query = input("Bạn hỏi: ").strip()
        if query.lower() in {"exit", "quit", ""}:
            break
        results = collection.query(query_texts=[query], n_results=1)
        if not results["documents"] or not results["documents"][0]:
            print("Không tìm thấy câu trả lời phù hợp.")
            continue

        answer = results["documents"][0][0]
        distance = results["distances"][0][0]
        meta = results["metadatas"][0][0]
        source_question = meta.get(COLUMN_INFO["question"], "Không rõ")
        source_id = meta.get(COLUMN_INFO["id"], "N/A")

        print("-" * 50)
        print(f"Mã hồ sơ: {source_id}")
        print(f"Câu hỏi/tựa đề gần nhất: {source_question}")
        print(f"Câu trả lời/khả năng: {answer}")
        print(f"Khoảng cách: {distance:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    interactive_search()
