"""Evaluation-only active rediscovery in the isolated official-RDKit environment."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
from rdkit import Chem, RDLogger

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--code',type=Path,default=Path('/LocalData1/uslee/datasets/molgenbench-v4/official-code'))
    args=ap.parse_args();root=args.output
    sys.path.insert(0,str(args.code))
    from molgenbench.metrics.hitrate import HitRediscoverMetric
    metric=HitRediscoverMetric(fixStereoFrom3D=False)
    RDLogger.DisableLog('rdApp.*')
    references=json.loads((root/'evaluation_only_actives.json').read_text())
    queries=json.loads((root/'activity_queries.json').read_text())
    needed={(r['target'],r['series']) for r in queries}
    lookup=defaultdict(lambda:defaultdict(list))
    for r in references:
        group=(r['target'],r['series'])
        if group not in needed:continue
        for inchi in metric._enumerate_tautomer_and_partial_chirality(r['smiles']):
            lookup[group][inchi].append(r)
    output=[]
    for q in queries:
        mol=Chem.MolFromSmiles(q['smiles'])
        inchi=Chem.MolToInchi(mol) if mol is not None else None
        matches=lookup[(q['target'],q['series'])].get(inchi,[])
        output.append(dict(**q,known_active_rediscovered=bool(matches),
            matched_record_names=sorted({r['name'] for r in matches}),
            experimental_values=[dict(endpoint=r['endpoint'],unit=r['unit'],value=r['affinity']) for r in matches],
            unmatched_activity='unknown' if not matches else None))
    dest=root/'activity_matches.json';temp=dest.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(time_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        protocol='Official tautomer/partial chirality reference enumeration; query graph stereo already inferred from unchanged output XYZ by DIRMO',
        official_code=str(args.code),matches=output),ensure_ascii=False,indent=2)+'\n');temp.replace(dest)

if __name__=='__main__':main()
