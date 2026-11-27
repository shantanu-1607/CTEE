"""Analytical counterexamples, not training data or labels for the supplied ZIP."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def impedance(resistance=0.0, inductance=0.0, capacitance=None, frequency=50.0):
    omega = 2*math.pi*frequency
    return complex(resistance, omega*inductance-(0 if capacitance is None else 1/(omega*capacitance)))

def main():
    frequency, voltage = 50.0, 230.0
    base_l = 0.1
    resonant_c = 1/((2*math.pi*frequency)**2*base_l)
    cases = [
        ("R_equals_resonant_RLC", "R", 10., 0., None),
        ("R_equals_resonant_RLC", "RLC", 10., base_l, resonant_c),
        ("RL_equals_RLC", "RL", 10., base_l, None),
        ("RL_equals_RLC", "RLC", 10., 2*base_l, resonant_c),
        ("RC_equals_RLC", "RC", 10., 0., resonant_c),
        ("RC_equals_RLC", "RLC", 10., base_l, resonant_c/2),
        ("L_equals_LC", "L", 0., base_l, None),
        ("L_equals_LC", "LC", 0., 2*base_l, resonant_c),
        ("C_equals_LC", "C", 0., 0., resonant_c),
        ("C_equals_LC", "LC", 0., base_l, resonant_c/2),
    ]
    rows = []
    for pair, circuit, r, l, c in cases:
        z = impedance(r, l, c, frequency)
        current = voltage/abs(z)
        pf = z.real/abs(z)
        rows.append(dict(pair=pair, circuit=circuit, resistance_ohm=r, inductance_h=l, capacitance_f=c, frequency_hz=frequency, voltage_rms_v=voltage, impedance_real_ohm=z.real, impedance_imag_ohm=z.imag, current_rms_a=current, real_power_w=voltage*current*pf, displacement_pf=pf))
    keys = ["impedance_real_ohm", "impedance_imag_ohm", "current_rms_a", "real_power_w", "displacement_pf"]
    for first, second in zip(rows[::2], rows[1::2]):
        for key in keys:
            assert math.isclose(first[key],second[key],rel_tol=1e-12,abs_tol=1e-10), (first["pair"],key)
    with (ROOT/"identifiability_examples.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status = {"intended_circuit_labels":["R","L","C","RL","RC","RLC","LC"],"requested_behaviour_labels":["constant_voltage","constant_current","constant_impedance"],"behaviour_definition_confirmed":False,"verified_target_labels_in_archive":False,"circuit_classifier_trained":False,"behaviour_classifier_trained":False,"existing_accuracy_target":"building_identity_only","existing_accuracy_valid_for_requested_task":False,"analytical_matching_pairs_verified":5,"counterexamples_are_training_data":False,"decision":"Await verified labels or an agreed replacement data source, plus behaviour definition; do not manufacture ground truth."}
    (ROOT/"task_status.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(status,indent=2))

if __name__ == "__main__": main()
