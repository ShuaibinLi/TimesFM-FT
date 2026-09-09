# Production split lists

Place the frozen one-minute dataset's chronological whole-session lists here:

```text
dates-train.txt
dates-val.txt
dates-test.txt
```

Each file must contain unique, ascending `YYYYMMDD` values, one per line. The
three sets must be disjoint. Generate them from the landed 390-minute source
sessions; do not copy the archived 500 ms experiment lists implicitly.

The bundle preparer hashes each file into `manifest.json`, and training or
evaluation refuses a bundle whose dates or hash differ from its configured
list.
