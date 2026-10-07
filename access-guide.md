# Participant Access Guide

This page walks you through every account and access step needed before the
hackathon starts (**Sunday October 11 – Friday October 16, 2026**).

⚠️ **Start today.** Several steps below require approval from someone else
(a project leader, IDRIS staff) and can take a few days. Don't leave this
until Monday morning.

## Checklist

1. [GitHub — challenge material](#1-github--challenge-material)
2. [Agora — event chat](#2-agora--event-chat)
3. [CC-IN2P3 account](#3-cc-in2p3-account)
4. [eDARI account + project attachment](#4-edari-account--project-attachment)
5. [DALIA compute account & SSH key](#5-dalia-compute-account--ssh-key)
6. (Optional) [Weights & Biases account](#6-optional-weights--biases-account)

---

## 1. GitHub — challenge material

All lectures, challenge descriptions, starter notebooks and data samples
live in:

**https://github.com/aghribi/artifact-hackathon-2026**

Clone it or just browse it — no account request needed, the repo is public.
Come back here regularly: content is still being filled in before the event.

## 2. Agora — event chat

Join the event's chat/community space (announcements, Q&A with case
holders, team coordination):

**https://agora.artifact-network.org/join?c=4nrX7rgYaFn4UqQSc2sWuKlHvpnvejJ5**

## 3. CC-IN2P3 account

You'll need an account at CC-IN2P3 (Lyon) regardless of which challenge you
join — it gives you a login node (`cca.in2p3.fr`) that several challenges'
data and, importantly, the DALIA access path (step 5) rely on.

1. Go to **https://id.cc.in2p3.fr/** and create an account.
2. Once created, request association to the **m4cast** project.
3. Wait for approval, then confirm you can log in:
   ```
   ssh <your_login>@cca.in2p3.fr
   ```

## 4. eDARI account + project attachment

eDARI is the portal used to request compute resources at IDRIS (the DALIA
supercomputer). This is a separate account from CC-IN2P3.

1. **Create your eDARI account**
   - Go to [edari.fr](https://www.edari.fr/user/login) and click
     "Se connecter ou se créer un compte eDARI" (top right).
   - Log in via "Fédération Éducation-Recherche" (RENATER) with your
     institutional credentials, or register with your institutional email.
   - Validate your email address and finalize your profile.

2. **Attach to project `AD011018409`**
   - Log in to your eDARI dashboard.
   - Under "LISTE DES ACTIONS GÉNÉRALES POSSIBLES", click
     "Se rattacher à un dossier ayant obtenu des ressources".
   - Enter the project identifier **AD011018409**, then click
     "Demander le rattachement".
   - Wait for the Project Leader (PI) to approve your request — this is a
     manual step on their side, so ping the organizers if it's pending.

## 5. DALIA compute account & SSH key

DALIA (IDRIS) only accepts SSH connections from **declared, fixed IP
addresses** — dynamic home IPs, university Wi-Fi, and commercial VPNs are
systematically rejected.

**For this hackathon, you don't need your own institutional fixed IP.**
You'll reach DALIA by jumping through CC-IN2P3's `cca.in2p3.fr`, whose
address is the one declared to IDRIS. That means: generate your DALIA SSH
key *on `cca`* (not on your laptop), and always connect to DALIA from there.

1. **Request the account**
   - On the eDARI homepage, click "Faire une demande d'ouverture de compte
     ou consulter le statut de la demande", targeting **IDRIS-EXT**.
   - Sign and submit the account declaration form (electronically on eDARI,
     or as a signed PDF to [gestutil@idris.fr](mailto:gestutil@idris.fr)).
   - Where the form asks for your fixed IP: use `cca.in2p3.fr`'s address —
     check with the organizers for the exact IP/range to declare.

2. **Wait for the onboarding email from IDRIS** confirming your login
   (e.g. `udlxxxxxx`).

3. **Generate your key — on `cca`, not locally:**
   ```
   ssh <your_cc-in2p3_login>@cca.in2p3.fr
   ssh-keygen -t ed25519 -f ~/.ssh/<your_login>-DALIA -C "<your_login>-DALIA"
   ```

4. **Send only the public key** (`~/.ssh/<your_login>-DALIA.pub`) back to
   IDRIS by replying to their onboarding email.

5. **Once IDRIS confirms the key is deployed**, log in to DALIA from `cca`:
   ```
   ssh -i ~/.ssh/<your_login>-DALIA <your_login>@dalia.idris.fr
   ```

## 6. (Optional) Weights & Biases account

If you'd like experiment tracking during your challenge, register for a
free [Weights & Biases educational account](https://wandb.ai/site).

---

Questions or stuck on approvals? Ask in Agora or contact the organizers
(Adnan Ghribi — adnan.ghribi@ganil.fr).
