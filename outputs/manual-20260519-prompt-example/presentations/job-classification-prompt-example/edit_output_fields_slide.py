from __future__ import annotations

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


def resize_shape(
    shape: ET.Element,
    *,
    y: int | None = None,
    cy: int | None = None,
) -> None:
    xfrm = shape.find("./p:spPr/a:xfrm", NS)
    if xfrm is None:
        raise ValueError("shape has no transform")
    off = xfrm.find("./a:off", NS)
    ext = xfrm.find("./a:ext", NS)
    if off is None or ext is None:
        raise ValueError("shape transform is incomplete")
    if y is not None:
        off.set("y", str(y))
    if cy is not None:
        ext.set("cy", str(cy))


def reset_body_margins(shape: ET.Element) -> None:
    body_pr = shape.find("./p:txBody/a:bodyPr", NS)
    if body_pr is None:
        raise ValueError("shape has no bodyPr")
    body_pr.set("lIns", "127000")
    body_pr.set("tIns", "50800")
    body_pr.set("rIns", "127000")
    body_pr.set("bIns", "50800")
    body_pr.set("anchor", "t")


def add_run(paragraph: ET.Element, text: str, *, size: str, color: str, bold: bool = False) -> None:
    run = ET.SubElement(paragraph, q("a:r"))
    r_pr = ET.SubElement(run, q("a:rPr"), {"sz": size})
    if bold:
        r_pr.set("b", "1")
    solid = ET.SubElement(r_pr, q("a:solidFill"))
    ET.SubElement(solid, q("a:srgbClr"), {"val": color})
    t = ET.SubElement(run, q("a:t"))
    t.text = text


def set_detail_box(shape: ET.Element, title: str, rows: list[tuple[str, str]]) -> None:
    reset_body_margins(shape)
    tx_body = shape.find("./p:txBody", NS)
    if tx_body is None:
        raise ValueError("shape has no txBody")
    for paragraph in list(tx_body.findall("./a:p", NS)):
        tx_body.remove(paragraph)

    title_p = ET.SubElement(tx_body, q("a:p"))
    ET.SubElement(title_p, q("a:pPr")).append(ET.Element(q("a:buNone")))
    add_run(title_p, title, size="1100", color="0F172A", bold=True)

    for key, desc in rows:
        paragraph = ET.SubElement(tx_body, q("a:p"))
        p_pr = ET.SubElement(paragraph, q("a:pPr"))
        ET.SubElement(p_pr, q("a:spcBef")).append(ET.Element(q("a:spcPts"), {"val": "80"}))
        ET.SubElement(p_pr, q("a:buNone"))
        add_run(paragraph, f"{key}: ", size="900", color="0F172A", bold=True)
        add_run(paragraph, desc, size="900", color="64748B")


def main() -> None:
    workspace = Path("/Users/jk/projects/python/llm-categorizing/outputs/manual-20260519-prompt-example/presentations/job-classification-prompt-example")
    source = workspace / "output" / "hi_feedback_job_classification_prompt_example.pptx"
    target = workspace / "output" / "hi_feedback_job_classification_prompt_example_with_output_fields.pptx"
    slide_part = "ppt/slides/slide1.xml"

    with zipfile.ZipFile(source, "r") as zin:
        root = ET.fromstring(zin.read(slide_part))

        resize_shape(shape_by_id(root, "121"), cy=2095500)

        expanded_y = 6591300
        expanded_h = 876300
        for shape_id in ("124", "125", "126"):
            resize_shape(shape_by_id(root, shape_id), y=expanded_y, cy=expanded_h)

        set_detail_box(
            shape_by_id(root, "125"),
            "검수 판단",
            [
                ("needs_review", "True면 사람 검수"),
                ("confidence", "확신도 0~1"),
                ("ambiguity_reason", "중복·충돌 사유"),
            ],
        )
        set_detail_box(
            shape_by_id(root, "126"),
            "설명 근거",
            [
                ("reason", "LLM 선택 근거"),
                ("guardrail_reason", "보정·차단 사유"),
                ("diagnosis_*", "진단 근거 요약"),
            ],
        )

        resize_shape(shape_by_id(root, "127"), y=7686675, cy=304800)
        resize_shape(shape_by_id(root, "128"), y=7753350, cy=228600)
        resize_shape(shape_by_id(root, "129"), y=7753350, cy=228600)
        resize_shape(shape_by_id(root, "130"), y=8018625, cy=171450)
        resize_shape(shape_by_id(root, "131"), y=8018625, cy=171450)

        updated_slide = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                payload = updated_slide if item.filename == slide_part else zin.read(item.filename)
                zout.writestr(item, payload)

    print(target)


if __name__ == "__main__":
    main()
