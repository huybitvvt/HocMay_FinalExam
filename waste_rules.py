CLASS_NAMES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
]

VI_LABELS = {
    "battery": "Pin / rác nguy hại",
    "biological": "Rác hữu cơ",
    "brown-glass": "Thủy tinh nâu",
    "cardboard": "Bìa carton",
    "clothes": "Quần áo / vải",
    "green-glass": "Thủy tinh xanh",
    "metal": "Kim loại",
    "paper": "Giấy",
    "plastic": "Nhựa",
    "shoes": "Giày dép",
    "trash": "Rác khác",
    "white-glass": "Thủy tinh trắng",
}

DISPOSAL_GUIDE = {
    "battery": {
        "group": "Nguy hại",
        "bin": "Điểm thu gom pin/rác điện tử",
        "action": "Không bỏ chung rác sinh hoạt. Dán hai cực pin nếu có thể, gom khô ráo và đưa tới điểm thu hồi pin.",
        "impact": "Giảm nguy cơ kim loại nặng rò rỉ vào đất và nước.",
    },
    "biological": {
        "group": "Hữu cơ",
        "bin": "Thùng rác hữu cơ / ủ compost",
        "action": "Tách khỏi nhựa, kim loại. Có thể ủ compost nếu không lẫn dầu mỡ hoặc hóa chất.",
        "impact": "Có thể chuyển thành phân hữu cơ, giảm rác chôn lấp.",
    },
    "brown-glass": {
        "group": "Tái chế",
        "bin": "Thùng thủy tinh",
        "action": "Rửa sơ, tháo nắp kim loại/nhựa. Bọc an toàn nếu bị vỡ.",
        "impact": "Thủy tinh tái chế giúp tiết kiệm nguyên liệu và năng lượng nung chảy.",
    },
    "green-glass": {
        "group": "Tái chế",
        "bin": "Thùng thủy tinh",
        "action": "Rửa sơ, phân loại cùng thủy tinh. Không trộn với gốm sứ hoặc bóng đèn.",
        "impact": "Tái chế thủy tinh nhiều lần mà ít giảm chất lượng.",
    },
    "white-glass": {
        "group": "Tái chế",
        "bin": "Thùng thủy tinh",
        "action": "Rửa sơ, bỏ nắp và nhãn nếu dễ tháo. Cẩn thận với mảnh sắc.",
        "impact": "Giảm nhu cầu khai thác cát và nguyên liệu sản xuất thủy tinh.",
    },
    "cardboard": {
        "group": "Tái chế",
        "bin": "Thùng giấy/carton",
        "action": "Làm phẳng, giữ khô, loại bỏ băng keo lớn hoặc thức ăn bám dính.",
        "impact": "Tái chế carton giúp giảm khai thác gỗ và tiết kiệm nước.",
    },
    "paper": {
        "group": "Tái chế",
        "bin": "Thùng giấy",
        "action": "Giữ khô, tránh lẫn giấy ăn bẩn, giấy dính dầu hoặc giấy phủ nhựa.",
        "impact": "Tăng tỷ lệ tái chế giấy và giảm rác chôn lấp.",
    },
    "plastic": {
        "group": "Tái chế",
        "bin": "Thùng nhựa tái chế",
        "action": "Rửa sơ chai/hộp, ép dẹp nếu có thể. Kiểm tra ký hiệu nhựa tại địa phương.",
        "impact": "Giảm rác nhựa trôi ra môi trường và tăng hiệu quả tái chế.",
    },
    "metal": {
        "group": "Tái chế",
        "bin": "Thùng kim loại",
        "action": "Rửa sơ lon/hộp, ép gọn nếu an toàn. Tách khỏi rác hữu cơ.",
        "impact": "Tái chế kim loại tiết kiệm năng lượng đáng kể so với sản xuất mới.",
    },
    "clothes": {
        "group": "Tái sử dụng",
        "bin": "Điểm quyên góp/thu hồi vải",
        "action": "Nếu còn dùng được, giặt sạch và quyên góp. Nếu hỏng, gom riêng để tái chế vải.",
        "impact": "Kéo dài vòng đời sản phẩm dệt may và giảm phát thải từ sản xuất mới.",
    },
    "shoes": {
        "group": "Tái sử dụng",
        "bin": "Điểm quyên góp/thu hồi giày",
        "action": "Làm sạch, ghép đủ đôi. Nếu không còn dùng được, hỏi điểm thu hồi vật liệu.",
        "impact": "Giảm rác khó phân hủy và tận dụng lại vật liệu cao su/vải.",
    },
    "trash": {
        "group": "Rác khác",
        "bin": "Thùng rác còn lại",
        "action": "Nếu không tái chế được hoặc bị bẩn nặng, bỏ vào rác còn lại. Kiểm tra lại nếu có pin, hóa chất, thủy tinh vỡ.",
        "impact": "Phân loại đúng giúp không làm bẩn dòng rác tái chế.",
    },
}

GROUP_COLORS = {
    "Tái chế": "#1f8a5b",
    "Hữu cơ": "#6b8e23",
    "Nguy hại": "#b42318",
    "Tái sử dụng": "#0f766e",
    "Rác khác": "#525252",
}


def get_waste_advice(class_name: str) -> dict:
    guide = DISPOSAL_GUIDE.get(class_name, DISPOSAL_GUIDE["trash"])
    return {
        "class_name": class_name,
        "label": VI_LABELS.get(class_name, class_name),
        **guide,
        "color": GROUP_COLORS.get(guide["group"], "#525252"),
    }
