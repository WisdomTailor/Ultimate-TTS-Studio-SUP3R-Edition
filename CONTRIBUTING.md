# Contributing

## Keep your working branch updated from `main` (without pushing to `main`)

Yes — you can safely pull updates from `main` into your branch without sending your branch changes to `main`.

```bash
git checkout <your-branch>
git fetch origin
git rebase origin/main
```

This updates only the branch you have checked out.  
Nothing is pushed to `main` unless you explicitly run a push command that targets `main`.

If you are developing on two branches, run the same commands on each branch:

```bash
git checkout <branch-one>
git fetch origin
git rebase origin/main

git checkout <branch-two>
git fetch origin
git rebase origin/main
```
