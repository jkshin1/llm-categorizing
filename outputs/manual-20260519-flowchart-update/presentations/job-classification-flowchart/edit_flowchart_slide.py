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


EMU_PER_PX = 9525


def emu(px: float) -> str:
    return str(round(px * EMU_PER_PX))


def q(name: str) -> str:
    prefix, local = name.split(":")
    return f"{{{NS[prefix]}}}{local}"


def shape_by_id(root: ET.Element, shape_id: str) -> ET.Element:
    for shape in root.findall(".//p:sp", NS):
        c_nv_pr = shape.find("./p:nvSpPr/p:cNvPr", NS)
        if c_nv_pr is not None and c_nv_pr.get("id") == shape_id:
            return shape
    raise ValueError(f"shape id not found: {shape_id}")


def sp_tree(root: ET.Element) -> ET.Element:
    tree = root.find("./p:cSld/p:spTree", NS)
    if tree is None:
        raise ValueError("slide has no shape tree")
    return tree


def set_bbox(shape: ET.Element, x: float, y: float, w: float, h: float) -> None:
    xfrm = shape.find("./p:spPr/a:xfrm", NS)
    if xfrm is None:
        raise ValueError("shape has no transform")
    off = xfrm.find("./a:off", NS)
    ext = xfrm.find("./a:ext", NS)
    if off is None or ext is None:
        raise ValueError("shape transform is incomplete")
    off.set("x", emu(x))
    off.set("y", emu(y))
    ext.set("cx", emu(w))
    ext.set("cy", emu(h))


def set_fill(shape: ET.Element, color: str) -> None:
    sp_pr = shape.find("./p:spPr", NS)
    if sp_pr is None:
        raise ValueError("shape has no spPr")
    fill = sp_pr.find("./a:solidFill", NS)
    if fill is None:
        fill = ET.SubElement(sp_pr, q("a:solidFill"))
    srgb = fill.find("./a:srgbClr", NS)
    if srgb is None:
        srgb = ET.SubElement(fill, q("a:srgbClr"))
    srgb.set("val", color)


def clear_text(shape: ET.Element) -> ET.Element:
    tx_body = shape.find("./p:txBody", NS)
    if tx_body is None:
        raise ValueError("shape has no txBody")
    for paragraph in list(tx_body.findall("./a:p", NS)):
        tx_body.remove(paragraph)
    return tx_body


def add_text_run(
    paragraph: ET.Element,
    text: str,
    *,
    size: str,
    color: str,
    bold: bool = False,
) -> None:
    run = ET.SubElement(paragraph, q("a:r"))
    r_pr = ET.SubElement(run, q("a:rPr"), {"sz": size})
    if bold:
        r_pr.set("b", "1")
    solid = ET.SubElement(r_pr, q("a:solidFill"))
    ET.SubElement(solid, q("a:srgbClr"), {"val": color})
    t = ET.SubElement(run, q("a:t"))
    t.text = text


def set_single_text(
    shape: ET.Element,
    text: str,
    *,
    size: str,
    color: str,
    bold: bool = False,
    align: str | None = None,
) -> None:
    tx_body = clear_text(shape)
    paragraph = ET.SubElement(tx_body, q("a:p"))
    p_pr = ET.SubElement(paragraph, q("a:pPr"))
    if align:
        p_pr.set("algn", align)
    ET.SubElement(p_pr, q("a:buNone"))
    add_text_run(paragraph, text, size=size, color=color, bold=bold)


def set_multiline_text(
    shape: ET.Element,
    lines: list[str],
    *,
    size: str,
    color: str,
    bold_first: bool = False,
) -> None:
    tx_body = clear_text(shape)
    for index, line in enumerate(lines):
        paragraph = ET.SubElement(tx_body, q("a:p"))
        p_pr = ET.SubElement(paragraph, q("a:pPr"))
        if index:
            ET.SubElement(p_pr, q("a:spcBef")).append(ET.Element(q("a:spcPts"), {"val": "80"}))
        ET.SubElement(p_pr, q("a:buNone"))
        add_text_run(paragraph, line, size=size, color=color, bold=bold_first and index == 0)


def set_node(
    root: ET.Element,
    ids: tuple[str, str, str, str],
    *,
    x: float,
    y: float,
    title: str,
    body_lines: list[str],
    badge_fill: str,
    badge_text: str,
    badge_color: str,
) -> None:
    bg, num, title_id, body_id = (shape_by_id(root, item) for item in ids)
    width = 300
    height = 126
    set_bbox(bg, x, y, width, height)
    set_bbox(num, x + 18, y + 18, 42, 42)
    set_bbox(title_id, x + 70, y + 18, width - 88, 34)
    set_bbox(body_id, x + 24, y + 62, width - 48, 48)
    set_fill(num, badge_fill)
    set_single_text(num, badge_text, size="1400", color=badge_color, bold=True, align="ctr")
    set_single_text(title_id, title, size="1350", color="0F172A", bold=True)
    set_multiline_text(body_id, body_lines, size="950", color="64748B")


def make_textbox(
    shape_id: int,
    name: str,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    size: str,
    color: str,
    bold: bool = False,
) -> ET.Element:
    shape = ET.Element(q("p:sp"))
    nv = ET.SubElement(shape, q("p:nvSpPr"))
    ET.SubElement(nv, q("p:cNvPr"), {"id": str(shape_id), "name": name})
    ET.SubElement(nv, q("p:cNvSpPr"), {"txBox": "1"})
    ET.SubElement(nv, q("p:nvPr"))
    sp_pr = ET.SubElement(shape, q("p:spPr"))
    xfrm = ET.SubElement(sp_pr, q("a:xfrm"))
    ET.SubElement(xfrm, q("a:off"), {"x": emu(x), "y": emu(y)})
    ET.SubElement(xfrm, q("a:ext"), {"cx": emu(w), "cy": emu(h)})
    geom = ET.SubElement(sp_pr, q("a:prstGeom"), {"prst": "rect"})
    ET.SubElement(geom, q("a:avLst"))
    ET.SubElement(sp_pr, q("a:noFill"))
    tx_body = ET.SubElement(shape, q("p:txBody"))
    ET.SubElement(tx_body, q("a:bodyPr"), {"wrap": "square", "lIns": "0", "tIns": "0", "rIns": "0", "bIns": "0", "anchor": "ctr"})
    ET.SubElement(tx_body, q("a:lstStyle"))
    paragraph = ET.SubElement(tx_body, q("a:p"))
    p_pr = ET.SubElement(paragraph, q("a:pPr"), {"algn": "ctr"})
    ET.SubElement(p_pr, q("a:buNone"))
    add_text_run(paragraph, text, size=size, color=color, bold=bold)
    return shape


def main() -> None:
    workspace = Path("/Users/jk/projects/python/llm-categorizing/outputs/manual-20260519-flowchart-update/presentations/job-classification-flowchart")
    source = Path(
        "/Users/jk/projects/python/llm-categorizing/outputs/manual-20260519-prompt-example/"
        "presentations/job-classification-prompt-example/output/"
        "hi_feedback_job_classification_prompt_example_with_output_fields.pptx"
    )
    target = workspace / "output" / "hi_feedback_job_classification_system_flowchart.pptx"
    slide_part = "ppt/slides/slide2.xml"

    with zipfile.ZipFile(source, "r") as zin:
        root = ET.fromstring(zin.read(slide_part))

        set_single_text(
            shape_by_id(root, "101"),
            "실제 동작: 캐시 분기, LLM 2회 호출, 코드 검증/보정",
            size="2700",
            color="0F172A",
            bold=True,
        )
        set_single_text(
            shape_by_id(root, "102"),
            "row 1개 기준으로 cache miss일 때 Stage 1/2 LLM을 호출하고, 최종 확정은 taxonomy 검증과 guardrail이 담당합니다.",
            size="1350",
            color="475569",
        )

        step_ids = {
            1: ("103", "104", "105", "106"),
            2: ("107", "108", "109", "110"),
            3: ("111", "112", "113", "114"),
            4: ("115", "116", "117", "118"),
            5: ("119", "120", "121", "122"),
            6: ("123", "124", "125", "126"),
            7: ("127", "128", "129", "130"),
        }

        set_node(
            root,
            step_ids[1],
            x=64,
            y=250,
            title="입력 근거 구성",
            body_lines=["self_review + diagnosis", "EvidenceTerm hints"],
            badge_fill="DBEAFE",
            badge_text="1",
            badge_color="1D4ED8",
        )
        set_node(
            root,
            step_ids[2],
            x=400,
            y=250,
            title="캐시 게이트",
            body_lines=["hit: LLM 0회", "miss: 분류 진행"],
            badge_fill="D1FAE5",
            badge_text="2",
            badge_color="047857",
        )
        set_node(
            root,
            step_ids[7],
            x=1072,
            y=250,
            title="CSV·검수 분기",
            body_lines=["정상 결과 저장", "needs_review 분리"],
            badge_fill="D1FAE5",
            badge_text="7",
            badge_color="047857",
        )
        set_node(
            root,
            step_ids[3],
            x=64,
            y=415,
            title="LLM 1차 호출",
            body_lines=["중직무/소직무 선택", "taxonomy pair 후보"],
            badge_fill="EDE9FE",
            badge_text="3",
            badge_color="6D28D9",
        )
        set_node(
            root,
            step_ids[4],
            x=400,
            y=415,
            title="코드 후보 필터",
            body_lines=["선택 pair 하위", "최종 후보만 남김"],
            badge_fill="DBEAFE",
            badge_text="4",
            badge_color="1D4ED8",
        )
        set_node(
            root,
            step_ids[5],
            x=736,
            y=415,
            title="LLM 2차 호출",
            body_lines=["Device/단위/세부", "최종 row 선택"],
            badge_fill="EDE9FE",
            badge_text="5",
            badge_color="6D28D9",
        )
        set_node(
            root,
            step_ids[6],
            x=1072,
            y=415,
            title="코드 검증·보정",
            body_lines=["taxonomy 검증", "진단·공정 guardrail"],
            badge_fill="FEE2E2",
            badge_text="6",
            badge_color="B91C1C",
        )

        # Reframe the lower panel as role separation, not another sequence.
        set_bbox(shape_by_id(root, "131"), 64, 565, 1408, 230)
        set_bbox(shape_by_id(root, "132"), 90, 588, 160, 32)
        set_bbox(shape_by_id(root, "133"), 90, 632, 1300, 38)
        set_single_text(shape_by_id(root, "132"), "호출·검증 구조", size="1100", color="1D4ED8", bold=True, align="ctr")
        set_single_text(
            shape_by_id(root, "133"),
            "LLM은 선택하고, 코드는 후보 제한·보정·검수 분리를 담당합니다.",
            size="1900",
            color="0F172A",
            bold=True,
        )
        panel_items = [
            ("134", "135", "136", 90, "LLM 선택", ["Stage 1/2에서 후보 중 선택", "confidence·reason 초안 생성"], "8B5CF6"),
            ("137", "138", "139", 552, "코드 보정", ["후보 필터링·taxonomy 검증", "diagnosis/process guardrail"], "2563EB"),
            ("140", "141", "142", 1014, "검수 분리", ["낮은 confidence·중복 직무·오류", "needs_review=True"], "F59E0B"),
        ]
        for dot_id, title_id, body_id, x, title, body, color in panel_items:
            set_bbox(shape_by_id(root, dot_id), x, 695, 13, 13)
            set_fill(shape_by_id(root, dot_id), color)
            set_bbox(shape_by_id(root, title_id), x + 24, 688, 400, 28)
            set_bbox(shape_by_id(root, body_id), x + 24, 722, 408, 54)
            set_single_text(shape_by_id(root, title_id), title, size="1350", color="0F172A", bold=True)
            set_multiline_text(shape_by_id(root, body_id), body, size="1050", color="64748B")

        # Move footer down slightly after panel compression.
        set_bbox(shape_by_id(root, "143"), 64, 820, 266, 24)
        set_bbox(shape_by_id(root, "144"), 1386, 820, 86, 24)

        tree = sp_tree(root)
        arrows = [
            (300, "ArrowInputCache", 366, 294, 28, 32, "→", "2300"),
            (301, "ArrowCacheOutput", 704, 287, 300, 32, "cache hit →", "1200"),
            (302, "ArrowCacheMiss", 510, 383, 90, 28, "miss ↓", "1200"),
            (303, "ArrowStage1Filter", 366, 459, 28, 32, "→", "2300"),
            (304, "ArrowFilterStage2", 702, 459, 28, 32, "→", "2300"),
            (305, "ArrowStage2Verify", 1038, 459, 28, 32, "→", "2300"),
            (306, "ArrowVerifyOutput", 1210, 383, 112, 28, "통과 ↑", "1150"),
        ]
        for shape_id, name, x, y, w, h, text, size in arrows:
            tree.append(
                make_textbox(
                    shape_id,
                    name,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    text=text,
                    size=size,
                    color="64748B",
                    bold=True,
                )
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
