# 部署到 Hugging Face Spaces

本说明介绍如何将 MIDI Editor 部署到 [Hugging Face Spaces](https://huggingface.co/spaces)。

## 方式一：Docker Space（推荐）

项目已包含 Dockerfile，Spaces 会使用 Docker 构建并运行前端。

### 步骤

1. **在 Hugging Face 创建 Space**
   - 打开 [huggingface.co/new-space](https://huggingface.co/new-space)
   - **Space name**：例如 `midi-editor`
   - **License**：选 MIT 或与你仓库一致
   - **SDK**：选择 **Docker**
   - 创建 Space

2. **在 Space 的 README 顶部添加 YAML 配置**

   在 Space 仓库的 `README.md` 最上方加入（保留你原有的 README 内容在下面即可）：

   ```yaml
   ---
   title: MIDI Editor
   sdk: docker
   app_port: 7860
   ---
   ```

3. **把本仓库代码推送到该 Space**
   - 在本地克隆/关联你的 HF Space 仓库（或直接克隆当前项目）
   - 确保仓库根目录包含：
     - `Dockerfile`
     - `nginx.conf`
     - `package.json`、`package-lock.json`
     - `index.html`、`vite.config.ts`、`src/`、`public/` 等全部源码
   - 推送到 Space 所在分支（例如 `main`）：
     ```bash
     git remote add space https://huggingface.co/spaces/<你的用户名>/<Space 名称>
     git push space main
     ```
   - 若 Space 是从本 GitHub 仓库“从仓库创建”的，则在该仓库中 push 到对应分支即可。

4. **等待构建**
   - Spaces 会根据 Dockerfile 执行 `npm ci`、`npm run build`，并用 nginx 在 **7860** 端口提供静态资源。
   - 构建完成后，访问 `https://huggingface.co/spaces/<用户名>/<Space 名称>` 即可使用。

### 端口说明

- Hugging Face Spaces 默认对外暴露的端口是 **7860**，因此 `Dockerfile` 和 `nginx.conf` 中已配置应用监听 7860。
- 若你在 Space 设置里修改了端口，需同步修改 `nginx.conf` 中的 `listen` 和 Space README 中的 `app_port`。

---

## 方式二：静态 HTML Space（可选）

若希望使用“静态 HTML”类型的 Space（无 Docker、无自动休眠），需要先在本地或 CI 中构建，再把构建产物上传到另一个 Static Space：

1. 本地执行：`npm ci && npm run build`
2. 新建一个 **Static** 类型的 Space
3. 将 `dist/` 目录下的全部内容（如 `index.html`、`assets/` 等）上传到该 Space 仓库根目录
4. 每次更新时重新构建并上传 `dist/` 内容

此方式需要你自行维护构建与上传流程（例如用 GitHub Actions 推送到 HF）。

---

## 故障排查

- **构建失败**：在 Space 的 “Logs” 中查看 Docker 构建日志，确认 `npm run build` 是否报错。
- **页面空白**：检查浏览器控制台是否有 404 或路径错误；确认 Vite 的 `base` 为 `'/'`（默认即可）。
- **端口不符**：确认 README 中的 `app_port: 7860` 与 `nginx.conf` 中的 `listen 7860` 一致。

部署完成后，你的 MIDI Editor 会有一个公开的在线地址，例如：  
`https://huggingface.co/spaces/你的用户名/midi-editor`。
