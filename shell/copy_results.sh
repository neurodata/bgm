(cd bgm/overleaf && git pull)
rsync -r --max-size=49m ./bgm/results/figs ./bgm/overleaf
# python ./bgm/docs/unglue_variables.py
# rsync ./bgm/docs/glued_variables.txt ./bgm/overleaf
(cd bgm/overleaf && git add . && git commit -m 'update figures' && git push)