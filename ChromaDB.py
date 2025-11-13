# -*- coding: utf-8 -*-
import csv  
import json
from pathlib import Path 
from typing import Dict, Iterable, List, Tuple
from sentence_transformers import SentenceTransformer
# Cài đặc thư viện chroma nếu chưa có bằng cú pháp: pip install chromadb
import chromadb
from typing import Dict, Iterable, List, Tuple

BASE_DIR = Path(__file__).resolve().parent 
CSV_FILE = BASE_DIR / "resume_CLEANED.csv"  # đg dẫn tới file CSV
COLLECTION_NAME = qa_collection  # tên collection
MAX_BATCH_SIZE = 5000  # smaller than Chroma batch limit

ID_CANDIDATES = ("id", "person_id", "uid") # các tên cột khả dĩ cho ID
TITLE_CANDIDATES = ("title", "job", "profession", "position", "name")
SKILLS_CANDIDATES = ("skills", "skill")
ABILITIES_CANDIDATES = ("abilities", "ability", "description")
PROGRAM_CANDIDATES = ("program", "education", "degree", "major")


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
    required: bool = True,
) -> str | None:
    """
    Chọn tên cột thực tế trong CSV khớp với 1 trong các candidates.
    Nếu required là True và không tìm thấy, ném lỗi.
    """
    for candidate in candidates:
        normalized = _normalize_header(candidate)
        if normalized in available_headers:
            return available_headers[normalized]

    if required:
        raise ValueError(
            f"No valid column '{label}' found in CSV. "
            f"CSV needs one of the following columns: {', '.join(candidates)}")
    return None

# Đọc CSV và trả về các tài liệu, metadata, ids và thông tin cột
def load_csv_rows(
    file_path: Path,
) -> Tuple[List[str], List[Dict[str, str]], List[str], Dict[str, str]]:
    """
    Trả về:
    - documents: list text sẽ embed bằng SentenceTransformer\
    - metadatas: list metadata có 4 filed: title, skills, abilities, program
    - ids: list id cho từng document
    - column_info: map lại tên cột gốc trong CSV
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []
    ids: List[str] = []

    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        normalized_headers = {_normalize_header(name): name for name in fieldnames}

        #ID (có thể bắt buộc, sẽ auto nếu thiếu)
        id_field = _resolve_field(normalized_headers, ID_CANDIDATES, "id", required=False) or "id_auto"

        #Câu hỏi (bắt buộc)
        title_field = _resolve_field(normalized_headers, TITLE_CANDIDATES, "title", required=True)
        skills_field = _resolve_field(normalized_headers, SKILLS_CANDIDATES, "skills", required=False)
        abilities_field = _resolve_field(normalized_headers, ABILITIES_CANDIDATES, "abilities", required=False)
        program_field = _resolve_field(normalized_headers, PROGRAM_CANDIDATES, "program", required=False)

        for row_index, row in enumerate(reader):
            title = (row.get(title_field) or "").strip() if title_field else ""
            skills = (row.get(skills_field) or "").strip() if skills_field else ""
            abilities = (row.get(abilities_field) or "").strip() if abilities_field else ""
            program = (row.get(program_field) or "").strip() if program_field else ""

            #nếu tất cả trống thì bỏ qua
            if not any([title, skills, abilities, program]):
                continue

            raw_id = (row.get(id_field) or "").strip() if id_field != "id_auto" else ""
            doc_id = raw_id or f"row_{len(documents)}"

            #Văn bản để embed (có thể chỉnh sửa format)
            #index_text là sự kết hợp của các trường
            index_text = "\n".join(
               [
                f"Title: {title}" if title else "",
                f"Skills: {skills}" if skills else "",
                f"Abilities: {abilities}" if abilities else "",
                f"Program: {program}" if program else "",
               ]
            ).strip()

            #Metadata dùng cho truy vấn sau này
            mentadata = {
                "title": title,
                "skills": skills,
                "abilities": abilities,
                "program": program,
                "source_id": doc_id,
            }

            # Thêm vào danh sách
            documents.append(index_text)
            metadatas.append(mentadata)
            ids.append(doc_id)
    
    # Kiểm tra có dòng hợp lệ không
    if not documents:
        raise ValueError("There are no valid rows in the CSV after processing.")

    # Trả về kết quả
    column_info = {
        "id": id_field,
        "title": title_field or "",
        "skills": skills_field or "",
        "abilities": abilities_field or "",
        "program": program_field or "",
    }
    return documents, metadatas, ids, column_info

# Nạp dữ liệu từ CSV vào collection và in tóm tắt
docs, metas, doc_ids, COLUMN_INFO = load_csv_rows(CSV_FILE)
#khởi tạo model embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
#Tạo embedding cho toàn bộ documents
embeddings = model.encode(docs, convert_to_tensor=False, show_progress_bar=True)
#chuyển sang list để upsert và chroma
emb_list = embeddings.tolist()
#Upsert kèm embedding theo từng lô nhỏ để tránh vượt giới hạn batch
total_docs = len(doc_ids)
for start_idx in range(0, total_docs, MAX_BATCH_SIZE):
    end_idx = min(start_idx + MAX_BATCH_SIZE, total_docs)
    collection.upsert(
        documents=docs[start_idx:end_idx],
        metadatas=metas[start_idx:end_idx],
        ids=doc_ids[start_idx:end_idx],
        embeddings=emb_list[start_idx:end_idx],
    )


# Lưu trữ vĩnh viễn nếu client hỗ trợ
if hasattr(client, "persist"):
    client.persist()
print(
    f"Loaded {len(doc_ids)} word flow {CSV_FILE}. ")
print(
    "Mapped column -> "
    f"title='{COLUMN_INFO.get('title') or '(empty)'}', "
    f"skills='{COLUMN_INFO.get('skills') or '(empty)'}', "
    f"abilities='{COLUMN_INFO.get('abilities') or '(empty)'}', "
    f"program='{COLUMN_INFO.get('program') or '(empty)'}'"
    )

# Hàm search trả về top 5 kết quả
def search_top5(query: str) -> List[Dict[str, str]]:
    """Trả về tối đa 5 hồ sơ phù hợp nhất với truy vấn."""
    #Encode query thành vector
    q_emb = model.encode([query], convert_to_tensor=False)[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=5,
        include=["metadatas", "distances"],
    )
    items: List[Dict[str, str]] = []
    metas_list = (results.get("metadatas") or [[]])[0]
    distance_list = (results.get("distances") or [[]])[0]
    for idx, meta in enumerate(metas_list):
        distance = distance_list[idx] if idx < len(distance_list) else None
        items.append({
            "title": meta.get("title", ""),
            "skills": meta.get("skills", ""),
            "abilities": meta.get("abilities", ""),
            "program": meta.get("program", ""),
            "distance": distance,
        })
    return items

# Chế độ hỏi-đáp tương tác
def interactive_search():
    print("Enter question (type 'exit' to exit)")
    while True:
        query = input("Question: ").strip()
        if query.lower() in {"exit", "quit", ""}:
            break
        
        items = search_top5(query)
        if not items:
            print("No suitable answer found.")
            continue

        #in ra đúng format JSON
        print(json.dumps(items, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    interactive_search()

