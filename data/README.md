# Training data

The lyrics are protected by copyright and are therefore **not included** in this repository.

To reproduce the experiment, put one plain-text file per artist into this folder, e.g.:

```
data/
├── goelae.txt
├── mani_matter.txt
├── pluesch.txt
└── polo_hofer.txt
```

The original study used song lyrics in Bernese Swiss German by Mani Matter, Plüsch, Polo Hofer and Gölä (about 52,000 characters in total). Any Swiss German text works.

`lyrics_gen/data.py` normalises the text automatically: lowercase, umlauts written as `ae`/`oe`/`ue`, punctuation removed.
