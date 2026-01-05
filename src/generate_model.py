import xml.etree.ElementTree as etree

def find_geom_by_name(root, name):
    for geom in root.iter('geom'):
        if geom.get('name') == name:
            return geom
    return None

def find_body_by_name(root, name):
    for body in root.iter('body'):
        if body.get('name') == name:
            return body
    return None

def set_front_left_leg_size(root, front_left_leg_len = 0.9, front_left_ankle_len = 1.5):
    leg_geom = find_geom_by_name(root, 'front_left_leg_geom')
    leg_geom_fromto = f"0.0 0.0 0.0 {front_left_leg_len} {front_left_leg_len} 0.0"
    leg_geom.set('fromto', leg_geom_fromto)

    ankle_body = find_body_by_name(root, 'front_left_ankle_body')
    ankle_body_pos = f"{front_left_leg_len} {front_left_leg_len} 0.0"
    ankle_body.set('pos', ankle_body_pos)

    ankle_geom = find_geom_by_name(root, 'front_left_ankle_geom')
    ankle_geom_fromto = f"0.0 0.0 0.0 {front_left_ankle_len} {front_left_ankle_len} 0.0"
    ankle_geom.set('fromto', ankle_geom_fromto)

    foot_geom = find_geom_by_name(root, 'front_left_foot_geom')
    foot_geom_fromto = f"{front_left_ankle_len} {front_left_ankle_len} 0.0"
    foot_geom.set('pos', foot_geom_fromto)

def set_front_right_leg_size(root, front_left_leg_len = 0.9, front_left_ankle_len = 1.5):
    leg_geom = find_geom_by_name(root, 'front_right_leg_geom')
    leg_geom_fromto = f"0.0 0.0 0.0 -{front_left_leg_len} {front_left_leg_len} 0.0"
    leg_geom.set('fromto', leg_geom_fromto)

    ankle_body = find_body_by_name(root, 'front_right_ankle_body')
    ankle_body_pos = f"-{front_left_leg_len} {front_left_leg_len} 0.0"
    ankle_body.set('pos', ankle_body_pos)

    ankle_geom = find_geom_by_name(root, 'front_right_ankle_geom')
    ankle_geom_fromto = f"0.0 0.0 0.0 -{front_left_ankle_len} {front_left_ankle_len} 0.0"
    ankle_geom.set('fromto', ankle_geom_fromto)

    foot_geom = find_geom_by_name(root, 'front_right_foot_geom')
    foot_geom_fromto = f"-{front_left_ankle_len} {front_left_ankle_len} 0.0"
    foot_geom.set('pos', foot_geom_fromto)

def set_back_left_leg_size(root, back_left_leg_len = 0.7, back_left_ankle_len = 0.3):
    leg_geom = find_geom_by_name(root, 'back_left_leg_geom')
    leg_geom_fromto = f"0.0 0.0 0.0 -{back_left_leg_len} -{back_left_leg_len} 0.0"
    leg_geom.set('fromto', leg_geom_fromto)

    ankle_body = find_body_by_name(root, 'back_left_ankle_body')
    ankle_body_pos = f"-{back_left_leg_len} -{back_left_leg_len} 0.0"
    ankle_body.set('pos', ankle_body_pos)

    ankle_geom = find_geom_by_name(root, 'back_left_ankle_geom')
    ankle_geom_fromto = f"0.0 0.0 0.0 -{back_left_ankle_len} -{back_left_ankle_len} 0.0"
    ankle_geom.set('fromto', ankle_geom_fromto)

    foot_geom = find_geom_by_name(root, 'back_left_foot_geom')
    foot_geom_fromto = f"-{back_left_ankle_len} -{back_left_ankle_len} 0.0"
    foot_geom.set('pos', foot_geom_fromto)

def set_back_right_leg_size(root, back_right_leg_len = 0.7, back_right_ankle_len = 0.3):
    leg_geom = find_geom_by_name(root, 'back_right_leg_geom')
    leg_geom_fromto = f"0.0 0.0 0.0 {back_right_leg_len} -{back_right_leg_len} 0.0"
    leg_geom.set('fromto', leg_geom_fromto)

    ankle_body = find_body_by_name(root, 'back_right_ankle_body')
    ankle_body_pos = f"{back_right_leg_len} -{back_right_leg_len} 0.0"
    ankle_body.set('pos', ankle_body_pos)

    ankle_geom = find_geom_by_name(root, 'back_right_ankle_geom')
    ankle_geom_fromto = f"0.0 0.0 0.0 {back_right_ankle_len} -{back_right_ankle_len} 0.0"
    ankle_geom.set('fromto', ankle_geom_fromto)

    foot_geom = find_geom_by_name(root, 'back_right_foot_geom')
    foot_geom_fromto = f"{back_right_ankle_len} -{back_right_ankle_len} 0.0"
    foot_geom.set('pos', foot_geom_fromto)

def update_model(root, front_leg_len = 0.9, front_ankle_len = 1.5, back_leg_len = 0.7, back_ankle_len = 0.3):
    set_front_left_leg_size(root, front_leg_len, front_ankle_len)
    set_front_right_leg_size(root, front_leg_len, front_ankle_len)
    set_back_left_leg_size(root, back_leg_len, back_ankle_len)
    set_back_right_leg_size(root, back_leg_len, back_ankle_len)

def generate_model(base_model_path, res_model_path, front_leg_len = 0.9, front_ankle_len = 1.5, back_leg_len = 0.7, back_ankle_len = 0.3):
    xml_tree = etree.parse(base_model_path)
    root = xml_tree.getroot()
    update_model(root, front_leg_len, front_ankle_len, back_leg_len, back_ankle_len)

    xml_tree.write(res_model_path, encoding='utf-8', xml_declaration=True)

