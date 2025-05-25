def disconnect_der(net, der_name="MG2 - Solar DER"):
    idx = net.sgen[net.sgen.name == der_name].index[0]
    net.sgen.at[idx, 'in_service'] = False

def inject_line_fault(net, line_idx):
    net.line.at[line_idx, 'in_service'] = False
