# [CUSTOM CELL] Colab-style: top-level statements run as the cell.
print('[INFO] hello from cell')

# Optional: define `run(ctx, prev)` to receive context and forward data
# to later cells. If defined, it runs after the top-level statements.
#
# def run(ctx, prev):
#     return {'data': {'key': 'value'}}
