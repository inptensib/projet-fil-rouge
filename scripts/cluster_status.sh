#!/bin/bash
# Affiche l'état des ressources du cluster : GPUs, CPUs, RAM par nœud.
# Trié du plus disponible au moins disponible (GPU libres desc, puis CPU libres desc).

PARTS=$(sinfo -h -o "%P" | tr -d '*' | sort -u | tr '\n' ',' | sed 's/,$//')

sinfo -N -h -p "$PARTS" --Format="NodeList:15,Partition:12,StateLong:12,Gres:30,GresUsed:30,CPUsState:16,Memory:10,FreeMem:10" | python3 -c "
import sys, re
nodes = {}
for line in sys.stdin:
    p = line.split()
    if len(p) < 8: continue
    node, part, state, gres, gresused, cpus, mem, freemem = p[:8]
    m = re.search(r'gpu:(\d+)', gres)
    gpu_tot = int(m.group(1)) if m else 0
    m = re.search(r'gpu:[^:]+:(\d+)', gresused)
    gpu_used = int(m.group(1)) if m else 0
    gpu_free = gpu_tot - gpu_used
    c = cpus.split('/')
    cpu_idle, cpu_tot = int(c[1]), int(c[3])
    entry = (gpu_free, cpu_idle, node, part, state, gpu_tot, cpu_tot, int(mem), int(freemem))
    if node not in nodes or gpu_tot > nodes[node][5]:
        nodes[node] = entry

rows = sorted(nodes.values(), key=lambda r: (r[0], r[1]), reverse=True)

print(f\"{'Nœud':<15} {'Partition':<12} {'État':<12} {'GPU libre/tot':<16} {'CPU libre/tot':<16} {'RAM tot':<10} {'RAM libre'}\")
print('─' * 90)
for gpu_free, cpu_idle, node, part, state, gpu_tot, cpu_tot, mem, freemem in rows:
    gpu_str = f'{gpu_free} lib / {gpu_tot} tot' if gpu_tot > 0 else '—'
    cpu_str = f'{cpu_idle} lib / {cpu_tot} tot'
    ram  = f'{mem/1024:.1f} GB'
    free = f'{freemem/1024:.1f} GB'
    print(f'{node:<15} {part:<12} {state:<12} {gpu_str:<16} {cpu_str:<16} {ram:<10} {free}')
"
