# Submission compliance review — 14 November 2026

The current pipeline writes 199,319 ordered frame,label rows with H/A/D/X and reproduces
the expected hash in README.md. Its only match input is automatic tactical ATD. No video,
public possession labels, manually corrected tracking, or hand-created match labels are
read during inference. Learned coefficients and decoder/gap settings come from external
SkillCorner match 2017461. Published graded-half boundaries define where to write X;
there are no game-specific forced D intervals. The standard pitch size is 105 by 68 m.

The organizers' emailed clarification takes precedence over older repository wording.
Automatic format and plausibility checks are used. No provided data is sent to AI services
by this software. The visual summary is generated locally from predictions alone and is
not compared with the video or ground truth. It is not published with this source.

## Historical disclosure

Earlier project work evaluated variants and parameter sensitivity against the public
possession window, and created video overlays for review. That prototype and its output
were retired after the organizers' clarification. The replacement fits and tunes on an
external match, but this does not erase that earlier exposure. The organizers alone can
determine whether this remediation satisfies their rules; this review is not a guarantee
of eligibility. No historical score is used as a claim about the current submission.

External phase-derived labels differ from the challenge definition; fragmentation remains
a performance limitation. The challenge private score is unknown. Registering team members,
the team name, acceptance of terms, and one response per team must be confirmed by the team.
Name the uploaded CSV and document after the registered team. Markdown documents are
accepted by SUBMISSION.md. The public technical document uses relative links to this repo;
the method and reproduction instructions themselves are included in the document.
