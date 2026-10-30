# 노트북 공개키를 LIMO authorized_keys 에 등록 (한 번만). LIMO 비밀번호를 한 번 묻는다.
$opts = @('-o', 'HostKeyAlias=limo', '-o', 'StrictHostKeyChecking=accept-new')
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" |
    ssh @opts limo "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo KEY_OK"
ssh @opts -o BatchMode=yes limo "echo LOGIN_OK"
