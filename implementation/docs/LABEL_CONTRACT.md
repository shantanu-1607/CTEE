# Electrical label contract - version 1

This is a scoped **synthetic controlled-source benchmark**, not proof of appliance internals or hardware performance. The professor has not approved this interpretation yet. Implementation proceeds under this explicit research assumption.

## The actual circuit represented

`measured AC voltage V -> controlled series voltage Uc -> series R/L/C branch -> return`

The branch obeys `Z(w)=R + r_parasitic + j(wL - 1/(wC))`. Absent components are omitted. Tiny resistance is explicitly recorded as a numerical/physical parasitic, not counted as a named R component. Composition labels refer to this **known branch**, not the entire electronically controlled device.

In CZ, Uc=0. In CC, the controller chooses a settled gain so terminal RMS current equals a reference. In CP, it chooses gain so terminal active power equals a nonzero reference. Internally, `Vbranch = gain * V`, `Uc=(gain-1)*V`, and current flows through the branch and controller. The controller can exchange energy. Power balance is `Pterminal + Pcontroller = Pbranch`. It is an ideal four-quadrant behavioural emulator, not a simulated switching converter or a claim that a passive RLC branch can be CP.

There are **18 supported pairs**: all seven compositions for CC and CZ; R/RL/RC/RLC for CP. Pure reactive named branches are excluded from CP despite their very small parasitic loss. Unsupported pairs raise an error.

Measurements are settled sine/multisine records. Probe 0 is nominal 50 Hz, probes 1/2 change voltage by -/+5%, probe 3 deliberately excites 3rd/5th/7th harmonics, and probe 4 is an 80 Hz bench/simulation measurement. Any active excitation is simulation-only. Composition evidence for controlled cases is fitted within one multisine record: controller gain is constant within that record, so component presence is preserved while equivalent component values rescale. Combining impedance magnitudes across differently regulated probes as if gain were constant would be invalid.

## Independent labels and limitations

The simulator knows the branch and operating mode. Feature extraction receives only measured voltage/current, sample rate, probe identity, and known source frequency. It never receives component values, mode labels, file names, group IDs, or controller gain. Training/test splitting is by the parent physical branch so its modes, corruptions, and probe subsets cannot leak across splits.

This benchmark does not establish arbitrary-topology identifiability, unseen appliance performance, switched-converter dynamics, mains-safe probing, or worldwide novelty. A zero-state passive startup solver provides an additional optional measured segment. For controlled fixtures its gain is preset to the settled nominal value during startup: this represents a fixed-gain startup experiment, not CC/CP regulation throughout the transient. Six normalised startup features are extracted, including the availability mask. Training randomly omits this capture in 30% of enhanced-model records so a CSV may legitimately omit it. Dynamic features from a realistic closed-loop converter remain future work. State these boundaries in the report. High simulator accuracy can reflect a mathematically easy benchmark; analytical rules and a random forest are required comparators.

## Changes from the planning document

The declared series topology is retained. Instead of silently assuming a real CC/CP converter with known input-filter topology, this first implementation uses an explicit ideal controlled-source emulator with energy accounting. Dynamic converter modelling, ngspice validation, and physical experiments remain separate later validation tasks.

