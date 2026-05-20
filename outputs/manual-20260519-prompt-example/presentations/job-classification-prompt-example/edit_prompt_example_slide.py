from __future__ import annotations

import copy
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def q(name: str) -> str:
    prefix, local = name.split(":")
    return f"{{{NS[prefix]}}}{local}"


def shape_by_id(root: ET.Element, shape_id: str) -> ET.Element:
    for shape in root.findall(".//p:sp", NS):
        c_nv_pr = shape.find("./p:nvSpPr/p:cNvPr", NS)
        if c_nv_pr is not None and c_nv_pr.get("id") == shape_id:
            return shape
    raise ValueError(f"shape id not found: {shape_id}")


def set_single_text(shape: ET.Element, text: str) -> None:
    text_node = shape.find(".//a:t", NS)
    if text_node is None:
        raise ValueError("shape has no text node")
    text_node.text = text


def set_bullets(shape: ET.Element, lines: list[str], *, font_size: str = "1150") -> None:
    tx_body = shape.find("./p:txBody", NS)
    if tx_body is None:
        raise ValueError("shape has no txBody")

    existing_paragraphs = tx_body.findall("./a:p", NS)
    if not existing_paragraphs:
        raise ValueError("shape has no paragraphs")
    template_p_pr = existing_paragraphs[0].find("./a:pPr", NS)
    if template_p_pr is None:
        raise ValueError("shape paragraph has no pPr")
    template_r_pr = existing_paragraphs[0].find("./a:r/a:rPr", NS)
    if template_r_pr is None:
        raise ValueError("shape paragraph has no rPr")

    for paragraph in existing_paragraphs:
        tx_body.remove(paragraph)

    for line in lines:
        paragraph = ET.SubElement(tx_body, q("a:p"))
        paragraph.append(copy.deepcopy(template_p_pr))
        run = ET.SubElement(paragraph, q("a:r"))
        r_pr = copy.deepcopy(template_r_pr)
        r_pr.set("sz", font_size)
        run.append(r_pr)
        text = ET.SubElement(run, q("a:t"))
        text.text = line


def main() -> None:
    workspace = Path("/Users/jk/projects/python/llm-categorizing/outputs/manual-20260519-prompt-example/presentations/job-classification-prompt-example")
    source = workspace / "template-starter.pptx"
    target = workspace / "output" / "hi_feedback_job_classification_prompt_example.pptx"
    slide_part = "ppt/slides/slide3.xml"

    with zipfile.ZipFile(source, "r") as zin:
        slide_xml = zin.read(slide_part)
        root = ET.fromstring(slide_xml)

        set_single_text(
            shape_by_id(root, "203"),
            "실제 self_review에서 동사·산출물·검증 활동을 찾아 taxonomy 후보를 좁히는 방식입니다.",
        )
        set_single_text(shape_by_id(root, "308"), "프롬프트 단서 추출")
        set_bullets(
            shape_by_id(root, "309"),
            [
                "동사: 검토·수행·평가·Tuning·개발",
                "산출물: Base Line·Scheme·Process Flow",
                "검증 활동: Process Qual·공정 조건 평가·Low-k IMD 평가",
            ],
        )
        set_single_text(shape_by_id(root, "313"), "판단 적용 예시")
        set_bullets(
            shape_by_id(root, "314"),
            [
                "예문: DRAM MLM Module Process Qual 수행",
                "DRAM/MLM은 대상·단위 단서, 직접 결론 아님",
                "Qual·평가·Flow가 강하면 중직무=공정 우선",
            ],
        )

        updated_slide = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                payload = updated_slide if item.filename == slide_part else zin.read(item.filename)
                zout.writestr(item, payload)

    print(target)


if __name__ == "__main__":
    main()
