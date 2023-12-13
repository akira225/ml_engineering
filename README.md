# HW2
 
## Data versioning via [Git LFS](https://git-lfs.com/)
---

#### Official Installation [Guide](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing)
---
#### Command to tell git to use LFS in repo:
```
git lfs install
```

#### Tracking files in project with LFS:
```
git lfs track '*.csv'
```

#### How to add/update files:
```
git add <...>
git commit -m ...
```

#### How to get files from remote repo with LFS:
```
git pull
git lfs pull
```