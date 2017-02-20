def get_bottom_padding(height, d, num_nodes, base_y, y_offset):
    return base_y + height / 2 - ((num_nodes * d + (num_nodes - 1) * y_offset) / 2)