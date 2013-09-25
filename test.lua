function h2x(hiragana)
	local h = string.gsub(hiragana, "[%s　]", "");
	return h
end

print(h2x("こん にちわ　，せ\nか	い"))
